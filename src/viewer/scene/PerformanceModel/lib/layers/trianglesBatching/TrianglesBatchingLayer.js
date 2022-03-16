import {WEBGL_INFO} from "../../../../webglInfo.js";
import {ENTITY_FLAGS} from '../../ENTITY_FLAGS.js';
import {RENDER_PASSES} from '../../RENDER_PASSES.js';

import {math} from "../../../../math/math.js";
import {RenderState} from "../../../../webgl/RenderState.js";
import {ArrayBuf} from "../../../../webgl/ArrayBuf.js";
import {geometryCompressionUtils} from "../../../../math/geometryCompressionUtils.js";
import {getBatchingRenderers} from "./TrianglesBatchingRenderers.js";
import {TrianglesBatchingBuffer} from "./TrianglesBatchingBuffer.js";
import {quantizePositions, transformAndOctEncodeNormals} from "../../compression.js";
import * as uniquifyPositions from "./calculateUniquePositions.js";
import { rebucketPositions } from "./rebucketPositions.js";
import {
    ramStats,
    DataTextureState,
    DataTextureGenerator
} from "../DataTextureState.js"

// 12-bits allowed for object ids
const MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER = (1 << 12);

// 2048 is max data texture height
const MAX_DATA_TEXTURE_HEIGHT = (1 << 11);

const INDICES_EDGE_INDICES_ALIGNEMENT_SIZE = 8;

let _maxEdgePortions = 2;

let _lastCanCreatePortion = {
    positions: null,
    indices: null,
    edgeIndices: null,
    uniquePositions: null,
    uniqueIndices: null,
    uniqueEdgeIndices: null,
    buckets: null,
};

const tempMat4 = math.mat4();
const tempMat4b = math.mat4();
const tempVec4a = math.vec4([0, 0, 0, 1]);
const tempVec4b = math.vec4([0, 0, 0, 1]);
const tempVec4c = math.vec4([0, 0, 0, 1]);
const tempOBB3 = math.OBB3();

const tempUint8Array4 = new Uint8Array (4);

const tempVec3a = math.vec3();
const tempVec3b = math.vec3();
const tempVec3c = math.vec3();
const tempVec3d = math.vec3();
const tempVec3e = math.vec3();
const tempVec3f = math.vec3();
const tempVec3g = math.vec3();

let _numberOfLayers = 0;

/**
 * @private
 */
class TrianglesBatchingLayer {

    /**
     * @param model
     * @param cfg
     * @param cfg.autoNormals
     * @param cfg.layerIndex
     * @param cfg.positionsDecodeMatrix
     * @param cfg.maxGeometryBatchSize
     * @param cfg.origin
     * @param cfg.scratchMemory
     * @param cfg.solid
     */
    constructor(model, cfg) {

        console.log ("create batching layer");

        this._layerNumber = _numberOfLayers++;
        ramStats.numberOfLayers++;

        /**
         * State sorting key.
         * @type {string}
         */
        this.sortId = "TrianglesBatchingLayer" + (cfg.solid ? "-solid" : "-surface") + (cfg.autoNormals ? "-autonormals" : "-normals");

        /**
         * Index of this TrianglesBatchingLayer in {@link PerformanceModel#_layerList}.
         * @type {Number}
         */
        this.layerIndex = cfg.layerIndex;

        this._batchingRenderers = getBatchingRenderers(model.scene);
        this.model = model;
        this._buffer = new TrianglesBatchingBuffer(cfg.maxGeometryBatchSize);
        this._scratchMemory = cfg.scratchMemory;

        /**
         * @type {DataTextureState}
         */
        this._dataTextureState = new DataTextureState ();

        /**
         * @type {DataTextureGenerator}
         */
        this.dataTextureGenerator = new DataTextureGenerator();

        this._state = new RenderState({
            offsetsBuf: null,
            metallicRoughnessBuf: null,
            positionsDecodeMatrix: math.mat4(),
            textureState: this._dataTextureState,
        });

        // These counts are used to avoid unnecessary render passes
        this._numPortions = 0;
        this._numVisibleLayerPortions = 0;
        this._numTransparentLayerPortions = 0;
        this._numXRayedLayerPortions = 0;
        this._numSelectedLayerPortions = 0;
        this._numHighlightedLayerPortions = 0;
        this._numClippableLayerPortions = 0;
        this._numEdgesLayerPortions = 0;
        this._numPickableLayerPortions = 0;
        this._numCulledLayerPortions = 0;

        this._modelAABB = math.collapseAABB3(); // Model-space AABB
        this._portions = [];

        this._numVerts = 0;

        this._numUniqueVerts = 0;

        this._numIndicesInLayer8Bits = 0;
        this._numIndicesInLayer16Bits = 0;
        this._numIndicesInLayer32Bits = 0;
        this._numEdgeIndicesInLayer8Bits = 0;
        this._numEdgeIndicesInLayer16Bits = 0;
        this._numEdgeIndicesInLayer32Bits = 0;

        this._finalized = false;

        if (cfg.positionsDecodeMatrix) {
            this._state.positionsDecodeMatrix.set(cfg.positionsDecodeMatrix);
            this._preCompressed = true;
        } else {
            this._preCompressed = false;
        }

        this._objectDataPositionsMatrices = []; // chipmunk
        this._objectDataColors = [];
        this._objectDataPickColors = [];

        this._vertexBasesForObject = []; // chipmunk

        this._portionIdForIndices8Bits = []; // chipmunk
        this._portionIdForIndices16Bits = []; // chipmunk
        this._portionIdForIndices32Bits = []; // chipmunk
        this._portionIdForEdges8Bits = []; // chipmunk
        this._portionIdForEdges16Bits = []; // chipmunk
        this._portionIdForEdges32Bits = []; // chipmunk

        this._portionIdFanOut = [];

        if (cfg.origin) {
            this._state.origin = math.vec3(cfg.origin);
        }

        /**
         * The axis-aligned World-space boundary of this TrianglesBatchingLayer's positions.
         * @type {*|Float64Array}
         */
        this.aabb = math.collapseAABB3();

        /**
         * When true, this layer contains solid triangle meshes, otherwise this layer contains surface triangle meshes
         * @type {boolean}
         */
        this.solid = !!cfg.solid;

        if (!this.model.cameraTexture)
        {
            this.model.cameraTexture = this.dataTextureGenerator.generateCameraDataTexture (
                this.model.scene.canvas.gl,
                this.model.scene.camera,
                this.model.scene,
                this._state.origin
            );
        }

        this._dataTextureState.textureCameraMatrices = this.model.cameraTexture;
    }

    /**
     * Tests if there is room for another portion in this TrianglesBatchingLayer.
     *
     * @param lenPositions Number of positions we'd like to create in the portion.
     * @param lenIndices Number of indices we'd like to create in this portion.
     * @returns {boolean} True if OK to create another portion.
     */
    canCreatePortion(positions, indices, edgeIndices) {
        if (this._finalized) {
            throw "Already finalized";
        }
        
        let uniquePositions, uniqueIndices, uniqueEdgeIndices;

        [
            uniquePositions,
            uniqueIndices,
            uniqueEdgeIndices,
        ] = uniquifyPositions.uniquifyPositions ({
            positions,
            indices,
            edgeIndices
        });

        let numUniquePositions = uniquePositions.length / 3;
        let numIndices = indices.length / 3;

        let buckets = rebucketPositions (
            {
                positions: uniquePositions,
                indices: uniqueIndices,
                edgeIndices: uniqueEdgeIndices,
            },
            (numUniquePositions > (1<< 16)) ? 16 : 8,
            // true
        );

        _lastCanCreatePortion.buckets = buckets;

        const newPortions = buckets ? buckets.length : 1;

        if ((this._numPortions + newPortions) > MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER)
        {
            ramStats.cannotCreatePortion.because10BitsObjectId++;
        }

        const maxIndicesOfAnyBits = Math.max (
            this._numIndicesInLayer8Bits,
            this._numIndicesInLayer16Bits,
            this._numIndicesInLayer32Bits,
        ) ;

        if (null !== buckets)
        {
            // debugger;

            numUniquePositions = 0;

            buckets.forEach(bucket => {
                numUniquePositions += bucket.positions.length / 3;
            });
        }

        if ((this._numUniqueVerts + numUniquePositions) > MAX_DATA_TEXTURE_HEIGHT * 512||
            (maxIndicesOfAnyBits + numIndices) > MAX_DATA_TEXTURE_HEIGHT * 512)
        {
            ramStats.cannotCreatePortion.becauseTextureSize++;
        }

        let retVal = (this._numPortions + newPortions) <= MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER && 
                     (this._numUniqueVerts + numUniquePositions) <= MAX_DATA_TEXTURE_HEIGHT * 512 &&
                     (maxIndicesOfAnyBits + numIndices) / 512 <= MAX_DATA_TEXTURE_HEIGHT * 512;

        if (!retVal)
        {
            // console.log ("Cannot create portion!");
            // console.log (this._numUniqueVerts + (_lastCanCreatePortion.uniquePositions.length / 3));
        }

        return retVal;
    }

    /**
     * Creates a new portion within this TrianglesBatchingLayer, returns the new portion ID.
     *
     * Gives the portion the specified geometry, color and matrix.
     *
     * @param cfg.positions Flat float Local-space positions array.
     * @param [cfg.normals] Flat float normals array.
     * @param [cfg.colors] Flat float colors array.
     * @param cfg.indices  Flat int indices array.
     * @param [cfg.edgeIndices] Flat int edges indices array.
     * @param cfg.color Quantized RGB color [0..255,0..255,0..255,0..255]
     * @param cfg.metallic Metalness factor [0..255]
     * @param cfg.roughness Roughness factor [0..255]
     * @param cfg.opacity Opacity [0..255]
     * @param [cfg.meshMatrix] Flat float 4x4 matrix
     * @param [cfg.worldMatrix] Flat float 4x4 matrix
     * @param cfg.worldAABB Flat float AABB World-space AABB
     * @param cfg.pickColor Quantized pick color
     * @returns {number} Portion ID
     */
    createPortion(cfg) {
        if (this._finalized) {
            throw "Already finalized";
        }

        if (cfg.indices == null)
        {
            return;
        }

        let buckets = _lastCanCreatePortion.buckets;

        if (buckets == null)
        {
            return;
        }

        let retVal = this._portionIdFanOut.length;

        this._portionIdFanOut.push ([]);

        buckets.forEach(bucket => {
            cfg.positions = bucket.positions;
            cfg.indices = bucket.indices;
            cfg.edgeIndices = bucket.edgeIndices;

            this._portionIdFanOut[retVal].push (
                this._createPortion (cfg)
            );
        });

        return retVal;
    }

    _createPortion(cfg) {
        ramStats.numberOfPortions++;

        // Indices alignement
        if (cfg.indices)
        {
            const alignedIndicesLen = Math.ceil ((cfg.indices.length / 3) / INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) * INDICES_EDGE_INDICES_ALIGNEMENT_SIZE * 3;

            ramStats.overheadSizeAlignementIndices += 2 * (alignedIndicesLen - cfg.indices.length);

            {
                const alignedIndices = new Uint32Array(alignedIndicesLen);
                alignedIndices.fill(0);
                alignedIndices.set (cfg.indices);
                cfg.indices = alignedIndices;
            }
        }

        // EdgeIndices alignement
        if (cfg.edgeIndices)
        {
            const alignedEdgeIndicesLen = Math.ceil ((cfg.edgeIndices.length / 2) / INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) * INDICES_EDGE_INDICES_ALIGNEMENT_SIZE * 2;

            ramStats.overheadSizeAlignementEdgeIndices += 2 * (alignedEdgeIndicesLen - cfg.edgeIndices.length);

            {
                const alignedEdgeIndices = new Uint32Array(alignedEdgeIndicesLen);
                alignedEdgeIndices.fill(0);
                alignedEdgeIndices.set (cfg.edgeIndices);
                cfg.edgeIndices = alignedEdgeIndices;
            }
        }
        
        this._numUniqueVerts += cfg.positions.length / 3;

        this._objectDataPositionsMatrices.push (this._positionsDecodeMatrix);

        const positions = cfg.positions;
        const normals = cfg.normals;
        const indices = cfg.indices;
        const edgeIndices = cfg.edgeIndices;
        const color = cfg.color;
        const metallic = cfg.metallic;
        const roughness = cfg.roughness;
        const colors = cfg.colors;
        const opacity = cfg.opacity;
        const meshMatrix = cfg.meshMatrix;
        const worldMatrix = cfg.worldMatrix;
        const worldAABB = cfg.worldAABB;
        const pickColor = cfg.pickColor;

        const scene = this.model.scene;
        const buffer = this._buffer;
        const positionsIndex = buffer.positions.length;
        const vertsIndex = positionsIndex / 3;
        const numVerts = positions.length / 3;
        const lenPositions = positions.length;

        if (this._preCompressed) {

            for (let i = 0, len = positions.length; i < len; i++) {
                buffer.positions.push(positions[i]);
            }

            const bounds = geometryCompressionUtils.getPositionsBounds(positions);

            const min = geometryCompressionUtils.decompressPosition(bounds.min, this._state.positionsDecodeMatrix, []);
            const max = geometryCompressionUtils.decompressPosition(bounds.max, this._state.positionsDecodeMatrix, []);

            worldAABB[0] = min[0];
            worldAABB[1] = min[1];
            worldAABB[2] = min[2];
            worldAABB[3] = max[0];
            worldAABB[4] = max[1];
            worldAABB[5] = max[2];

            if (worldMatrix) {
                math.AABB3ToOBB3(worldAABB, tempOBB3);
                math.transformOBB3(worldMatrix, tempOBB3);
                math.OBB3ToAABB3(tempOBB3, worldAABB);
            }

        } else {

            const positionsBase = buffer.positions.length;

            for (let i = 0, len = positions.length; i < len; i++) {
                buffer.positions.push(positions[i]);
            }

            if (meshMatrix) {

                for (let i = positionsBase, len = positionsBase + lenPositions; i < len; i += 3) {

                    tempVec4a[0] = buffer.positions[i + 0];
                    tempVec4a[1] = buffer.positions[i + 1];
                    tempVec4a[2] = buffer.positions[i + 2];

                    math.transformPoint4(meshMatrix, tempVec4a, tempVec4b);

                    buffer.positions[i + 0] = tempVec4b[0];
                    buffer.positions[i + 1] = tempVec4b[1];
                    buffer.positions[i + 2] = tempVec4b[2];

                    math.expandAABB3Point3(this._modelAABB, tempVec4b);

                    if (worldMatrix) {
                        math.transformPoint4(worldMatrix, tempVec4b, tempVec4c);
                        math.expandAABB3Point3(worldAABB, tempVec4c);
                    } else {
                        math.expandAABB3Point3(worldAABB, tempVec4b);
                    }
                }

            } else {

                for (let i = positionsBase, len = positionsBase + lenPositions; i < len; i += 3) {

                    tempVec4a[0] = buffer.positions[i + 0];
                    tempVec4a[1] = buffer.positions[i + 1];
                    tempVec4a[2] = buffer.positions[i + 2];

                    math.expandAABB3Point3(this._modelAABB, tempVec4a);

                    if (worldMatrix) {
                        math.transformPoint4(worldMatrix, tempVec4a, tempVec4b);
                        math.expandAABB3Point3(worldAABB, tempVec4b);
                    } else {
                        math.expandAABB3Point3(worldAABB, tempVec4a);
                    }
                }
            }
        }

        if (this._state.origin) {
            const origin = this._state.origin;
            worldAABB[0] += origin[0];
            worldAABB[1] += origin[1];
            worldAABB[2] += origin[2];
            worldAABB[3] += origin[0];
            worldAABB[4] += origin[1];
            worldAABB[5] += origin[2];
        }

        math.expandAABB3(this.aabb, worldAABB);

        if (colors) {
            // start of chipmunk
            this._objectDataColors.push ([
                colors[0] * 255,
                colors[1] * 255,
                colors[2] * 255,
                255
            ]);
        this._colorsLength = (this._colorsLength || 0) + colors.length * 4;
        // end of chipmunk

        } else if (color) {
            const r = color[0]; // Color is pre-quantized by PerformanceModel
            const g = color[1];
            const b = color[2];
            const a = opacity;

            // start of chipmunk
            this._objectDataColors.push ([
                r,
                g,
                b,
                opacity
            ]);
            // end of chipmunk
        }

        this._vertexBasesForObject.push (vertsIndex); // chupmunk

        const numUniquePositions = positions.length / 3;

        if (indices) {
            let triangleNumber = 0;
            for (let i = 0, len = indices.length; i < len; i+=3) {
                if (numUniquePositions <= (1<< 8)) {
                    buffer.indices8Bits.push(indices[i]);
                    buffer.indices8Bits.push(indices[i+1]);
                    buffer.indices8Bits.push(indices[i+2]);
                } else if (numUniquePositions <= (1<< 16)) {
                    buffer.indices16Bits.push(indices[i]);
                    buffer.indices16Bits.push(indices[i+1]);
                    buffer.indices16Bits.push(indices[i+2]);
                } else {
                    buffer.indices32Bits.push(indices[i]);
                    buffer.indices32Bits.push(indices[i+1]);
                    buffer.indices32Bits.push(indices[i+2]);
                }

                if ((triangleNumber % INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) == 0) {
                    if (numUniquePositions <= (1<< 8)) {
                        this._portionIdForIndices8Bits.push (this._numPortions);
                    } else if (numUniquePositions <= (1<< 16)) {
                        this._portionIdForIndices16Bits.push (this._numPortions);
                    }
                    else {
                        this._portionIdForIndices32Bits.push (this._numPortions);
                    }
                }
                triangleNumber++;
            }

            if (numUniquePositions <= (1<< 8)) {
                ramStats.totalPolygons8Bits += indices.length / 3;
                this._numIndicesInLayer8Bits += indices.length; // chupmunk
            } else if (numUniquePositions <= (1<< 16)) {
                ramStats.totalPolygons16Bits += indices.length / 3;
                this._numIndicesInLayer16Bits += indices.length; // chupmunk
            } else {
                ramStats.totalPolygons32Bits += indices.length / 3;
                this._numIndicesInLayer32Bits += indices.length; // chupmunk
            }

            ramStats.totalPolygons += indices.length / 3;
        }

        if (edgeIndices) {
            {
                const idealBytesPerIndex = Math.log2(numUniquePositions) / 8;

                ramStats.idealEdgeIndicesSize = (ramStats.idealEdgeIndicesSize || 0) + Math.max (
                    idealBytesPerIndex * edgeIndices.length,
                    1
                );
            }
            {
                if (numUniquePositions <= (1<< 8)) {
                    ramStats.edges8BitsSpace = (ramStats.edges8BitsSpace || 0) + edgeIndices.length;
                } else if (numUniquePositions <= (1<< 16)) {
                    ramStats.edges16BitsSpace = (ramStats.edges16BitsSpace || 0) + edgeIndices.length * 2;
                } else {
                    ramStats.edges32BitsSpace = (ramStats.edges32BitsSpace || 0) + edgeIndices.length * 4;
                }

                ramStats.optimizedEdgesSpace = (ramStats.edges8BitsSpace || 0) + (ramStats.edges16BitsSpace || 0) + (ramStats.edges32BitsSpace || 0);
                ramStats.nonOptimizedEdgesSpace = (ramStats.nonOptimizedEdgesSpace || 0) + edgeIndices.length * 2;
                ramStats.optimizedEdgesSavings = (ramStats.nonOptimizedEdgesSpace - ramStats.optimizedEdgesSpace);
            }

            let edgeNumber = 0;
            for (let i = 0, len = edgeIndices.length; i < len; i+=2) {
                if (numUniquePositions <= (1<< 8)) {
                    buffer.edgeIndices8Bits.push(edgeIndices[i]);
                    buffer.edgeIndices8Bits.push(edgeIndices[i+1]);
                } else if (numUniquePositions <= (1<< 16)) {
                    buffer.edgeIndices16Bits.push(edgeIndices[i]);
                    buffer.edgeIndices16Bits.push(edgeIndices[i+1]);
                } else {
                    buffer.edgeIndices32Bits.push(edgeIndices[i]);
                    buffer.edgeIndices32Bits.push(edgeIndices[i+1]);
                }

                if ((edgeNumber % INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) == 0) {
                    if (numUniquePositions <= (1<< 8)) {
                        this._portionIdForEdges8Bits.push (this._numPortions);
                    } else if (numUniquePositions <= (1<< 16)) {
                        this._portionIdForEdges16Bits.push (this._numPortions);
                    }
                    else {
                        this._portionIdForEdges32Bits.push (this._numPortions);
                    }
                }
                edgeNumber++;
            }
            if (numUniquePositions <= (1<< 8)) {
                this._numEdgeIndicesInLayer8Bits += indices.length; // chupmunk
            } else if (numUniquePositions <= (1<< 16)) {
                this._numEdgeIndicesInLayer16Bits += indices.length; // chupmunk
            } else {
                this._numEdgeIndicesInLayer32Bits += indices.length; // chupmunk
            }
        }

        // start of chipmunk
        this._objectDataPickColors.push (
            pickColor
        );
        // end of chipmunk

        // if (scene.entityOffsetsEnabled) {
        //     for (let i = 0; i < numVerts; i++) {
        //         buffer.offsets.push(0);
        //         buffer.offsets.push(0);
        //         buffer.offsets.push(0);
        //     }
        // }

        const portionId = this._portions.length;

        const portion = {
            vertsBase: vertsIndex,
            numVerts: numVerts
        };

        if (scene.pickSurfacePrecisionEnabled) {
            // Quantized in-memory positions are initialized in finalize()
            if (indices) {
                portion.indices = indices;
            }
            if (scene.entityOffsetsEnabled) {
                portion.offset = new Float32Array(3);
            }
        }

        this._portions.push(portion);

        this._numPortions++;
        this.model.numPortions++;

        this._numVerts += portion.numVerts;

        _lastCanCreatePortion = {
            buckets: null,
        };

        return portionId;
    }

    /**
     * Builds batch VBOs from appended geometries.
     * No more portions can then be created.
     */
    finalize() {
        if (this._finalized) {
            this.model.error("Already finalized");
            return;
        }

        const state = this._state;
        const textureState = this._dataTextureState;
        const gl = this.model.scene.canvas.gl;
        const buffer = this._buffer;

        state.gl = gl;

        // Generate all the needed textures in the layer

        // a) colors and flags texture
        textureState.texturePerObjectIdColorsAndFlags = this.dataTextureGenerator.generateTextureForColorsAndFlags (
            gl,
            this._objectDataColors,
            this._objectDataPickColors,
            this._vertexBasesForObject
        );

        // b) positions decode matrices texture
        textureState.texturePerObjectIdPositionsDecodeMatrix = this.dataTextureGenerator.generateTextureForPositionsDecodeMatrices (
            gl,
            this._objectDataPositionsMatrices
        ); 

        // c) position coordinates texture
        textureState.texturePerVertexIdCoordinates = this.dataTextureGenerator.generateTextureForPositions (
            gl,
            buffer.positions
        );

        // d) portion Id triangles texture
        textureState.texturePerPolygonIdPortionIds8Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices8Bits
        );

        textureState.texturePerPolygonIdPortionIds16Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices16Bits
        );

        textureState.texturePerPolygonIdPortionIds32Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices32Bits
        );

        // e) portion Id texture for edges
        textureState.texturePerEdgeIdPortionIds8Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges8Bits
        );

        textureState.texturePerEdgeIdPortionIds16Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges16Bits
        );

        textureState.texturePerEdgeIdPortionIds32Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges32Bits
        );

        // f) indices texture
        textureState.texturePerPolygonIdIndices8Bits = this.dataTextureGenerator.generateTextureFor8BitIndices (
            gl,
            buffer.indices8Bits
        );

        textureState.texturePerPolygonIdIndices16Bits = this.dataTextureGenerator.generateTextureFor16BitIndices (
            gl,
            buffer.indices16Bits
        );

        textureState.texturePerPolygonIdIndices32Bits = this.dataTextureGenerator.generateTextureFor32BitIndices (
            gl,
            buffer.indices32Bits
        );
        
        // g) edge indices texture
        textureState.texturePerPolygonIdEdgeIndices8Bits = this.dataTextureGenerator.generateTextureFor8BitsEdgeIndices (
            gl,
            buffer.edgeIndices8Bits
        );
        
        textureState.texturePerPolygonIdEdgeIndices16Bits = this.dataTextureGenerator.generateTextureFor16BitsEdgeIndices (
            gl,
            buffer.edgeIndices16Bits
        );
        
        textureState.texturePerPolygonIdEdgeIndices32Bits = this.dataTextureGenerator.generateTextureFor32BitsEdgeIndices (
            gl,
            buffer.edgeIndices32Bits
        );
        
        // end of chipmunk

        // if (buffer.metallicRoughness.length > 0) {
        //     const metallicRoughness = new Uint8Array(buffer.metallicRoughness);
        //     let normalized = false;
        //     state.metallicRoughnessBuf = new ArrayBuf(gl, gl.ARRAY_BUFFER, metallicRoughness, buffer.metallicRoughness.length, 2, gl.STATIC_DRAW, normalized);
        // }

        // if (this.model.scene.entityOffsetsEnabled) {
        //     if (buffer.offsets.length > 0) {
        //         const offsets = new Float32Array(buffer.offsets);
        //         state.offsetsBuf = new ArrayBuf(gl, gl.ARRAY_BUFFER, offsets, buffer.offsets.length, 3, gl.DYNAMIC_DRAW);
        //     }
        // }

        state.numIndices8Bits = buffer.indices8Bits.length;
        state.numIndices16Bits = buffer.indices16Bits.length;
        state.numIndices32Bits = buffer.indices32Bits.length;

        state.numEdgeIndices8Bits = buffer.edgeIndices8Bits.length;
        state.numEdgeIndices16Bits = buffer.edgeIndices16Bits.length;
        state.numEdgeIndices32Bits = buffer.edgeIndices32Bits.length;

        // Model matrices texture
        if (!this.model._modelMatricesTexture)
        {
            this.model._modelMatricesTexture = this.dataTextureGenerator.generatePeformanceModelDataTexture (
                gl, this.model
            );
        }

        textureState.textureModelMatrices = this.model._modelMatricesTexture;

        ramStats.additionalTheoreticalOptimalIndicesSavings = Math.round (
            (ramStats.sizeDataTextureIndices + ramStats.sizeDataTextureEdgeIndices) -
            (ramStats.idealIndicesSize + ramStats.idealEdgeIndicesSize)
        );

        // console.log (JSON.stringify(ramStats, null, 4));

        // let totalRamSize = 0;

        // Object.keys(ramStats).forEach (key => {
        //     if (key.startsWith ("size")) {
        //         totalRamSize+=ramStats[key];
        //     }
        // });

        // console.log (`Total size ${totalRamSize} bytes (${(totalRamSize/1000/1000).toFixed(2)} MB)`);
        // console.log (`Avg bytes / triangle: ${(totalRamSize / ramStats.totalPolygons).toFixed(2)}`);

        // let percentualRamStats = {};

        // Object.keys(ramStats).forEach (key => {
        //     if (key.startsWith ("size")) {
        //         percentualRamStats[key] = 
        //             `${(ramStats[key] / totalRamSize * 100).toFixed(2)} % of total`;
        //     }
        // });

        // console.log (JSON.stringify({percentualRamUsage: percentualRamStats}, null, 4));

        this._buffer = null;
        this._finalized = true;

        console.log (ramStats);

        _lastCanCreatePortion.buckets = null;
    }
        
    isEmpty() {
        return this._numPortions == 0;
    }

    initFlags(portionId, flags, meshTransparent) {
        if (flags & ENTITY_FLAGS.VISIBLE) {
            this._numVisibleLayerPortions++;
            this.model.numVisibleLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.HIGHLIGHTED) {
            this._numHighlightedLayerPortions++;
            this.model.numHighlightedLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.XRAYED) {
            this._numXRayedLayerPortions++;
            this.model.numXRayedLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.SELECTED) {
            this._numSelectedLayerPortions++;
            this.model.numSelectedLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.CLIPPABLE) {
            this._numClippableLayerPortions++;
            this.model.numClippableLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.EDGES) {
            this._numEdgesLayerPortions++;
            this.model.numEdgesLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.PICKABLE) {
            this._numPickableLayerPortions++;
            this.model.numPickableLayerPortions++;
        }
        if (flags & ENTITY_FLAGS.CULLED) {
            this._numCulledLayerPortions++;
            this.model.numCulledLayerPortions++;
        }
        if (meshTransparent) {
            this._numTransparentLayerPortions++;
            this.model.numTransparentLayerPortions++;
        }
        const deferred = true;
        this._setFlags(portionId, flags, meshTransparent, deferred);
        this._setFlags2(portionId, flags, deferred);
    }

    flushInitFlags() {
        this._setDeferredFlags();
        this._setDeferredFlags2();
    }

    setVisible(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.VISIBLE) {
            this._numVisibleLayerPortions++;
            this.model.numVisibleLayerPortions++;
        } else {
            this._numVisibleLayerPortions--;
            this.model.numVisibleLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setHighlighted(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.HIGHLIGHTED) {
            this._numHighlightedLayerPortions++;
            this.model.numHighlightedLayerPortions++;
        } else {
            this._numHighlightedLayerPortions--;
            this.model.numHighlightedLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setXRayed(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.XRAYED) {
            this._numXRayedLayerPortions++;
            this.model.numXRayedLayerPortions++;
        } else {
            this._numXRayedLayerPortions--;
            this.model.numXRayedLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setSelected(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.SELECTED) {
            this._numSelectedLayerPortions++;
            this.model.numSelectedLayerPortions++;
        } else {
            this._numSelectedLayerPortions--;
            this.model.numSelectedLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setEdges(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.EDGES) {
            this._numEdgesLayerPortions++;
            this.model.numEdgesLayerPortions++;
        } else {
            this._numEdgesLayerPortions--;
            this.model.numEdgesLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setClippable(portionId, flags) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.CLIPPABLE) {
            this._numClippableLayerPortions++;
            this.model.numClippableLayerPortions++;
        } else {
            this._numClippableLayerPortions--;
            this.model.numClippableLayerPortions--;
        }
        this._setFlags2(portionId, flags);
    }

    /**
     * This will _start_ a "set-flags transaction".
     * 
     * After invoking this method, calling setFlags/setFlags2 will not update
     * the colors+flags texture but only store the new flags/flag2 in the
     * colors+flags texture.
     * 
     * After invoking this method, and when all desired setFlags/setFlags2 have
     * been called on needed portions of the layer, invoke `commitDeferredFlags`
     * to actually update the texture data.
     * 
     * In massive "set-flags" scenarios like VFC or LOD mechanisms, the combina-
     * tion of `beginDeferredFlags` + `commitDeferredFlags`brings a speed-up of
     * up to 80x when e.g. objects are massively (un)culled 🚀.
     */
    beginDeferredFlags ()
    {
        this._deferredSetFlagsActive = true;
    }

    /**
     * This will _commit_ a "set-flags transaction".
     * 
     * Invoking this method will update the colors+flags texture data with new
     * flags/flags2 set since the previous invocation of `beginDeferredFlags`.
     */
    commitDeferredFlags ()
    {
        this._deferredSetFlagsActive = false;

        if (!this._deferredSetFlagsDirty)
        {
            return;
        }

        this._deferredSetFlagsDirty = false;

        const gl = this.model.scene.canvas.gl;
        const textureState = this._dataTextureState;
        
        gl.bindTexture (gl.TEXTURE_2D, textureState.texturePerObjectIdColorsAndFlags._texture);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0, // level
            0, // xoffset
            0, // yoffset
            textureState.texturePerObjectIdColorsAndFlags._textureWidth, // width
            textureState.texturePerObjectIdColorsAndFlags._textureHeight, // width
            gl.RGBA_INTEGER,
            gl.UNSIGNED_BYTE,
            textureState.texturePerObjectIdColorsAndFlags._textureData
        );
    }

    setCulled(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        
        if (flags & ENTITY_FLAGS.CULLED) {
            this._numCulledLayerPortions+=this._portionIdFanOut[portionId].length;
            this.model.numCulledLayerPortions++;
        } else {
            this._numCulledLayerPortions-=this._portionIdFanOut[portionId].length;
            this.model.numCulledLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setCollidable(portionId, flags) {
        if (!this._finalized) {
            throw "Not finalized";
        }
    }

    setPickable(portionId, flags, transparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.PICKABLE) {
            this._numPickableLayerPortions++;
            this.model.numPickableLayerPortions++;
        } else {
            this._numPickableLayerPortions--;
            this.model.numPickableLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    setColor(portionId, color) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        const portionsIdx = portionId;
        const portion = this._portions[portionsIdx];
        const vertexBase = portion.vertsBase;
        const numVerts = portion.numVerts;
        const firstColor = vertexBase * 4;
        const lenColor = numVerts * 4;
        const tempArray = this._scratchMemory.getUInt8Array(lenColor);
        const r = color[0];
        const g = color[1];
        const b = color[2];
        const a = color[3];
        for (let i = 0; i < lenColor; i += 4) {
            tempArray[i + 0] = r;
            tempArray[i + 1] = g;
            tempArray[i + 2] = b;
            tempArray[i + 3] = a;
        }
        // TODO: migrate to texture updates
        // if (this._state.colorsBuf) {
        //     this._state.colorsBuf.setData(tempArray, firstColor, lenColor);
        // }
    }

    setTransparent(portionId, flags, transparent) {
        if (transparent) {
            this._numTransparentLayerPortions++;
            this.model.numTransparentLayerPortions++;
        } else {
            this._numTransparentLayerPortions--;
            this.model.numTransparentLayerPortions--;
        }
        this._setFlags(portionId, flags, transparent);
    }

    _setFlags(portionId, flags, deferred = false) {
        (this._portionIdFanOut[portionId] || []).forEach (fanOut => {
            this._fan_out_setFlags (fanOut, flags, deferred);
        });
    }

    _fan_out_setFlags(portionId, flags, transparent, deferred = false) {
        if (!this._finalized) {
            throw "Not finalized";
        }

        const visible = !!(flags & ENTITY_FLAGS.VISIBLE);
        const xrayed = !!(flags & ENTITY_FLAGS.XRAYED);
        const highlighted = !!(flags & ENTITY_FLAGS.HIGHLIGHTED);
        const selected = !!(flags & ENTITY_FLAGS.SELECTED);
        const edges = !!(flags & ENTITY_FLAGS.EDGES);
        const pickable = !!(flags & ENTITY_FLAGS.PICKABLE);
        const culled = !!(flags & ENTITY_FLAGS.CULLED);

        // Color

        let f0;
        if (!visible || culled || xrayed) { // Highlight & select are layered on top of color - not mutually exclusive
            f0 = RENDER_PASSES.NOT_RENDERED;
        } else {
            if (transparent) {
                f0 = RENDER_PASSES.COLOR_TRANSPARENT;
            } else {
                f0 = RENDER_PASSES.COLOR_OPAQUE;
            }
        }

        // Silhouette

        let f1;
        if (!visible || culled) {
            f1 = RENDER_PASSES.NOT_RENDERED;
        } else if (selected) {
            f1 = RENDER_PASSES.SILHOUETTE_SELECTED;
        } else if (highlighted) {
            f1 = RENDER_PASSES.SILHOUETTE_HIGHLIGHTED;
        } else if (xrayed) {
            f1 = RENDER_PASSES.SILHOUETTE_XRAYED;
        } else {
            f1 = RENDER_PASSES.NOT_RENDERED;
        }

        // Edges

        let f2 = 0;
        if (!visible || culled) {
            f2 = RENDER_PASSES.NOT_RENDERED;
        } else if (selected) {
            f2 = RENDER_PASSES.EDGES_SELECTED;
        } else if (highlighted) {
            f2 = RENDER_PASSES.EDGES_HIGHLIGHTED;
        } else if (xrayed) {
            f2 = RENDER_PASSES.EDGES_XRAYED;
        } else if (edges) {
            if (transparent) {
                f2 = RENDER_PASSES.EDGES_COLOR_TRANSPARENT;
            } else {
                f2 = RENDER_PASSES.EDGES_COLOR_OPAQUE;
            }
        } else {
            f2 = RENDER_PASSES.NOT_RENDERED;
        }

        // Pick

        let f3 = (visible && !culled && pickable) ? RENDER_PASSES.PICK : RENDER_PASSES.NOT_RENDERED;

        const textureState = this._dataTextureState;
        const gl = this.model.scene.canvas.gl;

        tempUint8Array4 [0] = f0;
        tempUint8Array4 [1] = f1;
        tempUint8Array4 [2] = f2;
        tempUint8Array4 [3] = f3;

        // object flags
        textureState.texturePerObjectIdColorsAndFlags._textureData.set (
            tempUint8Array4,
            portionId * 24 + 8
        );

        if (this._deferredSetFlagsActive)
        {
            this._deferredSetFlagsDirty = true;
            return;
        }

        gl.bindTexture (gl.TEXTURE_2D, textureState.texturePerObjectIdColorsAndFlags._texture);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0, // level
            2, // xoffset
            portionId, // yoffset
            1, // width
            1, //height
            gl.RGBA_INTEGER,
            gl.UNSIGNED_BYTE,
            tempUint8Array4
        );

        // gl.bindTexture (gl.TEXTURE_2D, null);
    }

    _setDeferredFlags() {
    }

    _setFlags2(portionId, flags, deferred = false) {
        (this._portionIdFanOut[portionId] || []).forEach (fanOut => {
            this._fan_out_setFlags2 (fanOut, flags, deferred);
        });
    }

    _fan_out_setFlags2(portionId, flags, deferred = false) {
        if (!this._finalized) {
            throw "Not finalized";
        }

        const clippable = !!(flags & ENTITY_FLAGS.CLIPPABLE) ? 255 : 0;

        const textureState = this._dataTextureState;
        const gl = this.model.scene.canvas.gl;

        tempUint8Array4 [0] = clippable;
        tempUint8Array4 [1] = 0;
        tempUint8Array4 [2] = 1;
        tempUint8Array4 [3] = 2;

        // object flags2
        textureState.texturePerObjectIdColorsAndFlags._textureData.set (
            tempUint8Array4,
            portionId * 24 + 12
        );
        
        if (this._deferredSetFlagsActive)
        {
            this._deferredSetFlagsDirty = true;
            return;
        }
        
        gl.bindTexture (gl.TEXTURE_2D, textureState.texturePerObjectIdColorsAndFlags._texture);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0, // level
            3, // xoffset
            portionId, // yoffset
            1, // width
            1, //height
            gl.RGBA_INTEGER,
            gl.UNSIGNED_BYTE,
            tempUint8Array4
        );

        // gl.bindTexture (gl.TEXTURE_2D, null);
    }

    _setDeferredFlags2() {
        return;
    }

    setOffset(portionId, offset) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (!this.model.scene.entityOffsetsEnabled) {
            this.model.error("Entity#offset not enabled for this Viewer"); // See Viewer entityOffsetsEnabled
            return;
        }
        const portionsIdx = portionId;
        const portion = this._portions[portionsIdx];
        const vertexBase = portion.vertsBase;
        const numVerts = portion.numVerts;
        const firstOffset = vertexBase * 3;
        const lenOffsets = numVerts * 3;
        const tempArray = this._scratchMemory.getFloat32Array(lenOffsets);
        const x = offset[0];
        const y = offset[1];
        const z = offset[2];
        for (let i = 0; i < lenOffsets; i += 3) {
            tempArray[i + 0] = x;
            tempArray[i + 1] = y;
            tempArray[i + 2] = z;
        }
        if (this._state.offsetsBuf) {
            this._state.offsetsBuf.setData(tempArray, firstOffset, lenOffsets);
        }
        if (this.model.scene.pickSurfacePrecisionEnabled) {
            portion.offset[0] = offset[0];
            portion.offset[1] = offset[1];
            portion.offset[2] = offset[2];
        }
    }

    // ---------------------- COLOR RENDERING -----------------------------------

    drawColorOpaque(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numTransparentLayerPortions === this._numPortions || this._numXRayedLayerPortions === this._numPortions) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (frameCtx.withSAO && this.model.saoEnabled) {
            if (frameCtx.pbrEnabled && this.model.pbrEnabled) {
                if (this._batchingRenderers.colorQualityRendererWithSAO) {
                    this._batchingRenderers.colorQualityRendererWithSAO.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            } else {
                if (this._batchingRenderers.colorRendererWithSAO) {
                    this._batchingRenderers.colorRendererWithSAO.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            }
        } else {
            if (frameCtx.pbrEnabled && this.model.pbrEnabled) {
                if (this._batchingRenderers.colorQualityRenderer) {
                    this._batchingRenderers.colorQualityRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            } else {
                if (this._batchingRenderers.colorRenderer) {
                    this._batchingRenderers.colorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            }
        }
    }

    _updateBackfaceCull(renderFlags, frameCtx) {
        const backfaces = this.model.backfaces || (!this.solid) || renderFlags.sectioned;
        if (frameCtx.backfaces !== backfaces) {
            const gl = frameCtx.gl;
            if (backfaces) {
                gl.disable(gl.CULL_FACE);
            } else {
                gl.enable(gl.CULL_FACE);
            }
            frameCtx.backfaces = backfaces;
        }
    }

    drawColorTransparent(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numTransparentLayerPortions === 0 || this._numXRayedLayerPortions === this._numPortions) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (frameCtx.pbrEnabled && this.model.pbrEnabled) {
            if (this._batchingRenderers.colorQualityRenderer) {
                this._batchingRenderers.colorQualityRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_TRANSPARENT);
            }
        } else {
            if (this._batchingRenderers.colorRenderer) {
                this._batchingRenderers.colorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_TRANSPARENT);
            }
        }
    }

    // ---------------------- RENDERING SAO POST EFFECT TARGETS --------------

    drawDepth(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numTransparentLayerPortions === this._numPortions || this._numXRayedLayerPortions === this._numPortions) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.depthRenderer) {
            this._batchingRenderers.depthRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE); // Assume whatever post-effect uses depth (eg SAO) does not apply to transparent objects
        }
    }

    drawNormals(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numTransparentLayerPortions === this._numPortions || this._numXRayedLayerPortions === this._numPortions) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.normalsRenderer) {
            this._batchingRenderers.normalsRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);  // Assume whatever post-effect uses normals (eg SAO) does not apply to transparent objects
        }
    }

    // ---------------------- SILHOUETTE RENDERING -----------------------------------

    drawSilhouetteXRayed(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numXRayedLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.silhouetteRenderer) {
            this._batchingRenderers.silhouetteRenderer.drawLayer(frameCtx, this, RENDER_PASSES.SILHOUETTE_XRAYED);
        }
    }

    drawSilhouetteHighlighted(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numHighlightedLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.silhouetteRenderer) {
            this._batchingRenderers.silhouetteRenderer.drawLayer(frameCtx, this, RENDER_PASSES.SILHOUETTE_HIGHLIGHTED);
        }
    }

    drawSilhouetteSelected(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numSelectedLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.silhouetteRenderer) {
            this._batchingRenderers.silhouetteRenderer.drawLayer(frameCtx, this, RENDER_PASSES.SILHOUETTE_SELECTED);
        }
    }

    // ---------------------- EDGES RENDERING -----------------------------------

    drawEdgesColorOpaque(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numEdgesLayerPortions === 0) {
            return;
        }
        if (this._batchingRenderers.edgesColorRenderer) {
            this._batchingRenderers.edgesColorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.EDGES_COLOR_OPAQUE);
        }
    }

    drawEdgesColorTransparent(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numEdgesLayerPortions === 0 || this._numTransparentLayerPortions === 0) {
            return;
        }
        if (this._batchingRenderers.edgesColorRenderer) {
            this._batchingRenderers.edgesColorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.EDGES_COLOR_TRANSPARENT);
        }
    }

    drawEdgesHighlighted(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numHighlightedLayerPortions === 0) {
            return;
        }
        if (this._batchingRenderers.edgesRenderer) {
            this._batchingRenderers.edgesRenderer.drawLayer(frameCtx, this, RENDER_PASSES.EDGES_HIGHLIGHTED);
        }
    }

    drawEdgesSelected(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numSelectedLayerPortions === 0) {
            return;
        }
        if (this._batchingRenderers.edgesRenderer) {
            this._batchingRenderers.edgesRenderer.drawLayer(frameCtx, this, RENDER_PASSES.EDGES_SELECTED);
        }
    }

    drawEdgesXRayed(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0 || this._numXRayedLayerPortions === 0) {
            return;
        }
        if (this._batchingRenderers.edgesRenderer) {
            this._batchingRenderers.edgesRenderer.drawLayer(frameCtx, this, RENDER_PASSES.EDGES_XRAYED);
        }
    }

    // ---------------------- OCCLUSION CULL RENDERING -----------------------------------

    drawOcclusion(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.occlusionRenderer) {
            this._batchingRenderers.occlusionRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
        }
    }

    // ---------------------- SHADOW BUFFER RENDERING -----------------------------------

    drawShadow(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.shadowRenderer) {
            this._batchingRenderers.shadowRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
        }
    }

    //---- PICKING ----------------------------------------------------------------------------------------------------

    drawPickMesh(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.pickMeshRenderer) {
            this._batchingRenderers.pickMeshRenderer.drawLayer(frameCtx, this, RENDER_PASSES.PICK);
        }
    }

    drawPickDepths(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.pickDepthRenderer) {
            this._batchingRenderers.pickDepthRenderer.drawLayer(frameCtx, this, RENDER_PASSES.PICK);
        }
    }

    drawPickNormals(renderFlags, frameCtx) {
        if (this._numCulledLayerPortions === this._numPortions || this._numVisibleLayerPortions === 0) {
            return;
        }
        this._updateBackfaceCull(renderFlags, frameCtx);
        if (this._batchingRenderers.pickNormalsRenderer) {
            this._batchingRenderers.pickNormalsRenderer.drawLayer(frameCtx, this, RENDER_PASSES.PICK);
        }
    }

    //------------------------------------------------------------------------------------------------

    precisionRayPickSurface(portionId, worldRayOrigin, worldRayDir, worldSurfacePos, worldNormal) {

        if (!this.model.scene.pickSurfacePrecisionEnabled) {
            return false;
        }

        const state = this._state;
        const portion = this._portions[portionId];

        if (!portion) {
            this.model.error("portion not found: " + portionId);
            return false;
        }

        const positions = portion.quantizedPositions;
        const indices = portion.indices;
        const origin = state.origin;
        const offset = portion.offset;

        const rtcRayOrigin = tempVec3a;
        const rtcRayDir = tempVec3b;

        rtcRayOrigin.set(origin ? math.subVec3(worldRayOrigin, origin, tempVec3c) : worldRayOrigin);  // World -> RTC
        rtcRayDir.set(worldRayDir);

        if (offset) {
            math.subVec3(rtcRayOrigin, offset);
        }

        math.transformRay(this.model.worldNormalMatrix, rtcRayOrigin, rtcRayDir, rtcRayOrigin, rtcRayDir); // RTC -> local

        const a = tempVec3d;
        const b = tempVec3e;
        const c = tempVec3f;

        let gotIntersect = false;
        let closestDist = 0;
        const closestIntersectPos = tempVec3g;

        for (let i = 0, len = indices.length; i < len; i += 3) {

            const ia = indices[i] * 3;
            const ib = indices[i + 1] * 3;
            const ic = indices[i + 2] * 3;

            a[0] = positions[ia];
            a[1] = positions[ia + 1];
            a[2] = positions[ia + 2];

            b[0] = positions[ib];
            b[1] = positions[ib + 1];
            b[2] = positions[ib + 2];

            c[0] = positions[ic];
            c[1] = positions[ic + 1];
            c[2] = positions[ic + 2];

            math.decompressPosition(a, state.positionsDecodeMatrix);
            math.decompressPosition(b, state.positionsDecodeMatrix);
            math.decompressPosition(c, state.positionsDecodeMatrix);

            if (math.rayTriangleIntersect(rtcRayOrigin, rtcRayDir, a, b, c, closestIntersectPos)) {

                math.transformPoint3(this.model.worldMatrix, closestIntersectPos, closestIntersectPos);

                if (offset) {
                    math.addVec3(closestIntersectPos, offset);
                }

                if (origin) {
                    math.addVec3(closestIntersectPos, origin);
                }

                const dist = Math.abs(math.lenVec3(math.subVec3(closestIntersectPos, worldRayOrigin, [])));

                if (!gotIntersect || dist > closestDist) {
                    closestDist = dist;
                    worldSurfacePos.set(closestIntersectPos);
                    if (worldNormal) { // Not that wasteful to eagerly compute - unlikely to hit >2 surfaces on most geometry
                        math.triangleNormal(a, b, c, worldNormal);
                    }
                    gotIntersect = true;
                }
            }
        }

        if (gotIntersect && worldNormal) {
            math.transformVec3(this.model.worldNormalMatrix, worldNormal, worldNormal);
            math.normalizeVec3(worldNormal);
        }

        return gotIntersect;
    }

    // ---------

    destroy() {
        const state = this._state;
        if (state.offsetsBuf) {
            state.offsetsBuf.destroy();
            state.offsetsBuf = null;
        }
        if (state.metallicRoughnessBuf) {
            state.metallicRoughnessBuf.destroy();
            state.metallicRoughnessBuf = null;
        }
        state.destroy();
    }
}

export {TrianglesBatchingLayer};