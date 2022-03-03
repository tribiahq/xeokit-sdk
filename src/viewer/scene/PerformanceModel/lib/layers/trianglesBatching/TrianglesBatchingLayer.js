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
import { Float16Array, isFloat16Array, getFloat16, setFloat16, hfround, } from "./float16.js";
import * as uniquifyPositions from "./calculateUniquePositions.js";
import { rebucketPositions } from "./rebucketPositions.js";
import { createRTCViewMat } from "../../../../math/rtcCoords.js";

// 12-bits allowed for object ids
const MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER = (1 << 12);

// 2048 is max data texture height
const MAX_DATA_TEXTURE_HEIGHT = (1 << 11);

const INDICES_EDGE_INDICES_ALIGNEMENT_SIZE = 8;

let _maxEdgePortions = 2;

let ramStats = {
    sizeDataColorsAndFlags: 0,
    sizeDataPositionDecodeMatrices: 0,
    sizeDataTexturePositions: 0,
    sizeDataTextureIndices: 0,
    sizeDataTextureEdgeIndices: 0,
    sizeDataTexturePortionIds: 0,
    numberOfPortions: 0,
    numberOfLayers: 0,
    totalPolygons: 0,
    totalPolygons8Bits: 0,
    totalPolygons16Bits: 0,
    totalPolygons32Bits: 0,
    cannotCreatePortion: {
        because10BitsObjectId: 0,
        becauseTextureSize: 0,
    },
    overheadSizeAlignementIndices: 0, 
    overheadSizeAlignementEdgeIndices: 0, 
};

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

let textureCameraMatrices = {
    lastOrigin: null,
    differentOrigin: function (origin)
    {
        if (this.lastOrigin == null)
        {
            return true;
        }

        for (let i = 0; i < 3; i++)
        {
            if (origin[i] != this.lastOrigin[i])
            {
                return true;
            }
        }

        return false;
    },
    lastCameraViewMatrix: null,
    differentCameraViewMatrix: function (viewMatrix)
    {
        if (this.lastCameraViewMatrix == null)
        {
            return true;
        }

        for (let i = 0; i < 16; i++)
        {
            if (viewMatrix[i] != this.lastCameraViewMatrix [i])
            {
                return true;
            }
        }

        return false;
    },
    lastCameraViewNormalMatrix: null,
    differentCameraViewNormalMatrix: function (viewNormalMatrix)
    {
        if (this.lastCameraViewNormalMatrix == null)
        {
            return true;
        }

        for (let i = 0; i < 16; i++)
        {
            if (viewNormalMatrix[i] != this.lastCameraViewNormalMatrix [i])
            {
                return true;
            }
        }

        return false;
    },
    lastCameraProjectMatrix: null,
    differentCameraProjectMatrix: function (projectMatrix)
    {
        if (this.lastCameraProjectMatrix == null)
        {
            return true;
        }

        for (let i = 0; i < 16; i++)
        {
            if (projectMatrix[i] != this.lastCameraProjectMatrix [i])
            {
                return true;
            }
        }

        return false;
    },
    ensureCameraMatrices: function (gl, origin, viewMatrix, viewNormalMatrix, projectMatrix)
    {
        if (this.differentOrigin(origin) || 
            this.differentCameraViewMatrix(viewMatrix))
        {
            this.lastOrigin = origin.slice ();
            this.lastCameraViewMatrix = viewMatrix.slice ();
            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                0,
                4,
                1,
                gl.RGBA,
                gl.FLOAT,
                ((origin) ? createRTCViewMat(viewMatrix, origin) : viewMatrix).slice ()
            );
            // console.log ("different1");
        }
        else
        {
            // console.log ("equal1");
        }

        if (this.differentCameraViewNormalMatrix (viewNormalMatrix))
        {
            this.lastCameraViewNormalMatrix = viewNormalMatrix.slice ();
            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                1,
                4,
                1,
                gl.RGBA,
                gl.FLOAT,
                new Float32Array (viewNormalMatrix)
            );
            // console.log ("different2");
        }
        else
        {
            // console.log ("equal2");
        }

        if (this.differentCameraProjectMatrix (projectMatrix))
        {
            this.lastCameraProjectMatrix = projectMatrix.slice ();
            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                2,
                4,
                1,
                gl.RGBA,
                gl.FLOAT,
                new Float32Array (projectMatrix)
            );
            // console.log ("different3");
        }
        else
        {
            // console.log ("equal3");
        }
    },
    texture: null
};

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

        this._dataTextureState = {
            /**
             * Texture that holds colors/pickColors/flags/flags2 per-object:
             * - columns: one concept per column => color / pick-color / ...
             * - row: the object Id
             */
            texturePerObjectIdColorsAndFlags: null,
            /**
             * The number of objects stored in `texturePerObjectIdColorsAndFlags`.
             */
            texturePerObjectIdColorsAndFlagsHeight: null,
            /**
             * Texture that holds the positionsDecodeMatrix per-object:
             * - columns: each column is one column of the matrix
             * - row: the object Id
             */
            texturePerObjectIdPositionsDecodeMatrix: null,
            /**
             * Texture that holds all the `different-vertices` used by the layer.
             */            
            texturePerVertexIdCoordinates: null,
            texturePerVertexIdCoordinatesHeight: null,
            /**
             * Texture that holds the PortionId that corresponds to a given polygon-id.
             * 
             * Variant of the texture for 8-bit based polygon-ids.
             */
            texturePerPolygonIdPortionIds8Bits: null,
            /**
             * Texture that holds the PortionId that corresponds to a given polygon-id.
             * 
             * Variant of the texture for 16-bit based polygon-ids.
             */
            texturePerPolygonIdPortionIds16Bits: null,
            /**
             * Texture that holds the PortionId that corresponds to a given polygon-id.
             * 
             * Variant of the texture for 32-bit based polygon-ids.
             */
            texturePerPolygonIdPortionIds32Bits: null,
            /**
             * Texture that holds the PortionId that corresponds to a given edge-id.
             * 
             * Variant of the texture for 8-bit based polygon-ids.
             */
            texturePerEdgeIdPortionIds8Bits: null,
            /**
             * Texture that holds the PortionId that corresponds to a given edge-id.
             * 
             * Variant of the texture for 16-bit based polygon-ids.
             */
            texturePerEdgeIdPortionIds16Bits: null,
            /**
             * Texture that holds the PortionId that corresponds to a given edge-id.
             * 
             * Variant of the texture for 32-bit based polygon-ids.
             */
            texturePerEdgeIdPortionIds32Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 8-bit based indices.
             */            
            texturePerPolygonIdIndices8Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 16-bit based indices.
             */            
            texturePerPolygonIdIndices16Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 32-bit based indices.
             */            
            texturePerPolygonIdIndices32Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 8-bit based edge indices.
             */            
            texturePerPolygonIdEdgeIndices8Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 16-bit based edge indices.
             */            
            texturePerPolygonIdEdgeIndices16Bits: null,
            /**
             * Texture that holds the unique-vertex-indices for 32-bit based edge indices.
             */            
            texturePerPolygonIdEdgeIndices32Bits: null,
            /**
             * Texture that holds the camera matrices
             * - columns: each column in the texture is a camera matrix column.
             * - row: each row is a different camera matrix.
             */
            textureCameraMatrices: null,
            /**
             * Texture that holds the model matrices
             * - columns: each column in the texture is a model matrix column.
             * - row: each row is a different model matrix.
             */
            textureModelMatrices: null,
        };

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
            const camera = this.model.scene.camera;
            const scene = this.model.scene;

            const gl = this.model.scene.canvas.gl;
            
            const textureWidth = 4;
            const textureHeight = 3; // space for 3 matrices

            const texture = gl.createTexture();

            gl.bindTexture (gl.TEXTURE_2D, texture);
            
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, textureWidth, textureHeight);

            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
            gl.bindTexture (gl.TEXTURE_2D, null);

            this.model.cameraTexture = this.generateBindableTexture(
                gl,
                texture,
                textureWidth,
                textureHeight
            );

            const self = this;

            let cameraDirty = true;

            const onCameraMatrix = () => {
                if (!cameraDirty) {
                    return;
                }

                cameraDirty = false;
                
                gl.bindTexture (gl.TEXTURE_2D, self.model.cameraTexture._texture);

                const origin = self._state.origin;

                // Camera's "view matrix"
                gl.texSubImage2D(
                    gl.TEXTURE_2D,
                    0,
                    0,
                    0,
                    4,
                    1,
                    gl.RGBA,
                    gl.FLOAT,
                    new Float32Array ((origin) ? createRTCViewMat(camera.viewMatrix, origin) : camera.viewMatrix)
                );
    
                // Camera's "view normal matrix"
                gl.texSubImage2D(
                    gl.TEXTURE_2D,
                    0,
                    0,
                    1,
                    4,
                    1,
                    gl.RGBA,
                    gl.FLOAT,
                    new Float32Array (camera.viewNormalMatrix)
                );

                // Camera's "project matrix"
                gl.texSubImage2D(
                    gl.TEXTURE_2D,
                    0,
                    0,
                    2,
                    4,
                    1,
                    gl.RGBA,
                    gl.FLOAT,
                    new Float32Array (camera.project.matrix)
                );
            };

            camera.on ("matrix", () => cameraDirty = true);

            scene.on ("rendering", onCameraMatrix);

            onCameraMatrix ();
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

        if ((cfg.positions.length / 3) > (1<<16))
        {
            console.log (`YAY! ${(cfg.positions.length / 3)} positions`);
        }

        // Indices 64-triangles aglignement
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

        // EdgeIndices 64-edged alignement
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

        for (let i = 0; i < numVerts; i++)
        {
            buffer.objectData.push (this._numPortions); // chipmunk
        }

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
                // buffer.indices.push(indices[i]);
                // buffer.indices.push(indices[i+1]);

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
                // buffer.edgeIndices.push(edgeIndices[i]);
                // buffer.edgeIndices.push(edgeIndices[i+1]);
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
        textureState.texturePerObjectIdColorsAndFlags = this.generateTextureForColorsAndFlags (
            gl,
            this._objectDataColors,
            this._objectDataPickColors,
            this._vertexBasesForObject
        );

        // b) positions decode matrices texture
        textureState.texturePerObjectIdPositionsDecodeMatrix = this.generateTextureForPositionsDecodeMatrices (
            gl,
            this._objectDataPositionsMatrices
        ); 

        // c) position coordinates texture
        textureState.texturePerVertexIdCoordinates = this.generateTextureForPositions (
            gl,
            buffer.positions
        );

        // d) portion Id triangles texture
        textureState.texturePerPolygonIdPortionIds8Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices8Bits
        );

        textureState.texturePerPolygonIdPortionIds16Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices16Bits
        );

        textureState.texturePerPolygonIdPortionIds32Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices32Bits
        );

        // e) portion Id texture for edges
        textureState.texturePerEdgeIdPortionIds8Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges8Bits
        );

        textureState.texturePerEdgeIdPortionIds16Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges16Bits
        );

        textureState.texturePerEdgeIdPortionIds32Bits = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges32Bits
        );

        // f) indices texture
        textureState.texturePerPolygonIdIndices8Bits = this.generateTextureFor8BitIndices (
            gl,
            buffer.indices8Bits
        );

        textureState.texturePerPolygonIdIndices16Bits = this.generateTextureFor16BitIndices (
            gl,
            buffer.indices16Bits
        );

        textureState.texturePerPolygonIdIndices32Bits = this.generateTextureFor32BitIndices (
            gl,
            buffer.indices32Bits
        );
        
        // g) edge indices texture
        textureState.texturePerPolygonIdEdgeIndices8Bits = this.generateTextureFor8BitsEdgeIndices (
            gl,
            buffer.edgeIndices8Bits
        );
        
        textureState.texturePerPolygonIdEdgeIndices16Bits = this.generateTextureFor16BitsEdgeIndices (
            gl,
            buffer.edgeIndices16Bits
        );
        
        textureState.texturePerPolygonIdEdgeIndices32Bits = this.generateTextureFor32BitsEdgeIndices (
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
            const textureWidth = 4;
            const textureHeight = 2; // space for 2 matrices

            const texture = gl.createTexture();

            gl.bindTexture (gl.TEXTURE_2D, texture);
            
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, textureWidth, textureHeight);

            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0, // x-offset
                0, // y-offset (model world matrix)
                4, // data width (4x4 values)
                1, // data height (1 matrix)
                gl.RGBA,
                gl.FLOAT,
                new Float32Array (this.model.worldMatrix.slice ())
            );

            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0, // x-offset
                1, // y-offset (model normal matrix)
                4, // data width (4x4 values)
                1, // data height (1 matrix)
                gl.RGBA,
                gl.FLOAT,
                new Float32Array (this.model.worldNormalMatrix.slice ())
            );

            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
            gl.bindTexture (gl.TEXTURE_2D, null);

            this.model._modelMatricesTexture = this.generateBindableTexture(
                gl,
                texture,
                textureWidth,
                textureHeight
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

        _lastCanCreatePortion.buckets = null;
    }

    generateBindableTexture (gl, texture, textureWidth, textureHeight, textureData = null)
    {
        return {
            /**
             * The WebGLRenderingContext.
             * @private
             */
            _gl: gl,
            /**
             * The WebGLTexture handle.
             * @private
             */
            _texture: texture,
            /**
             * The texture width.
             * @private
             */
            _textureWidth: textureWidth,
            /**
             * The texture height.
             * @private
             */
            _textureHeight: textureHeight,
            /**
             * Then the texture data array is kept in the JS side, it will be stored here.
             * @private
             */
            _textureData: textureData,
            /**
             * Convenience method to be used by the renderers to bind the texture before draw calls.
             * @public
             */
            bindTexture: function (glProgram, shaderName, glTextureUnit) {
                return glProgram.bindTexture (shaderName, this, glTextureUnit);
            },
            /**
             * Used internally by the `program` passed to `bindTexture` in order to bind the texture to an active `texture-unit`.
             * @private
             */
            bind: function (unit) {
                this._gl.activeTexture(this._gl["TEXTURE" + unit]);
                this._gl.bindTexture(this._gl.TEXTURE_2D, this._texture);
                return true;
            },
            /**
             * Used internally by the `program` passed to `bindTexture` in order to bind the texture to an active `texture-unit`.
             * @private
             */
            unbind: function (unit) {
                // This `unbind` method is ignored at the moment to allow avoiding to rebind same texture already bound to a texture unit.

                // this._gl.activeTexture(this.state.gl["TEXTURE" + unit]);
                // this._gl.bindTexture(this.state.gl.TEXTURE_2D, null);
            }
        };
    }

    /**
     * This will generate an RGBA texture for:
     * - colors
     * - pickColors
     * - flags
     * - flags2
     * 
     * The texture will have:
     * - 4 RGBA columns per row: for each object (pick) color and flags(2)
     * - N rows where N is the number of objects
     * 
     * @param {*} gl WebGL2Context 
     * @param {*} colors Array of colors for all objects in the layer
     * @param {*} pickColors Array of pickColors for all objects in the layer
     * 
     * @returns The created texture and its height
     */
    generateTextureForColorsAndFlags (gl, colors, pickColors, vertexBases) {
        // The number of rows in the texture is the number of
        // objects in the layer.

        const textureHeight = colors.length;

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        // 4 columns per texture row:
        // - col0: (RGBA) object color RGBA
        // - col1: (packed Uint32 as RGBA) object pick color
        // - col2: (packed 4 bytes as RGBA) object flags
        // - col3: (packed 4 bytes as RGBA) object flags2
        const textureWidth = 6;

        const texArray = new Uint8Array (4 * textureWidth * textureHeight);

        ramStats.sizeDataColorsAndFlags +=texArray.byteLength;

        for (var i = 0; i < textureHeight; i++)
        {
            // object color
            texArray.set (
                colors [i],
                i * 24 + 0
            );

            // object pick color
            texArray.set (
                pickColors [i],
                i * 24 + 4
            );

            // object flags
            texArray.set (
                [
                    0, 0, 0, 0
                ],
                i * 24 + 8
            );

            // object flags2
            texArray.set (
                [
                    0, 0, 0, 0
                ],
                i * 24 + 12
            );

            // vertex base
            texArray.set (
                [
                    (vertexBases[i] >> 24) & 255,
                    (vertexBases[i] >> 16) & 255,
                    (vertexBases[i] >> 8) & 255,
                    (vertexBases[i]) & 255,
                ],
                i * 24 + 16
            );
        }

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGBA_INTEGER,
            gl.UNSIGNED_BYTE,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight,
            texArray
        );
    }

    /**
     * This will generate a texture for all positions decode matrices in the layer.
     * 
     * The texture will have:
     * - 4 RGBA columns per row (each column will contain 4 packed half-float (16 bits) components).
     *   Thus, each row will contain 16 packed half-floats corresponding to a complete positions decode matrix)
     * - N rows where N is the number of objects
     * 
     * @param {*} gl WebGL2Context 
     * @param {*} positionDecodeMatrices Array of positions decode matrices for all objects in the layer
     * 
     * @returns The created texture and its height
     */
    generateTextureForPositionsDecodeMatrices (gl, positionDecodeMatrices) {
        const textureHeight =  positionDecodeMatrices.length;

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const textureWidth = 4;

        var texArray = new Float16Array(4 * textureWidth * textureHeight);

        ramStats.sizeDataPositionDecodeMatrices +=texArray.byteLength;

        for (var i = 0; i < positionDecodeMatrices.length; i++)
        {
            // 4 values
            texArray.set (
                positionDecodeMatrices [i],
                i * 16
            );
        }

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);
        
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGBA,
            gl.HALF_FLOAT,
            new Uint16Array (texArray.buffer),
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: document
     */
    generateTextureFor8BitIndices (gl, indices) {
        if (indices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }

        const textureWidth = 512;
        const textureHeight = Math.ceil (indices.length / 3 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 3;
        const texArray = new Uint8Array (texArraySize);

        ramStats.sizeDataTextureIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(indices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB8UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGB_INTEGER,
            gl.UNSIGNED_BYTE,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: document
     */
    generateTextureFor16BitIndices (gl, indices) {
        if (indices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }
        const textureWidth = 512;
        const textureHeight = Math.ceil (indices.length / 3 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 3;
        const texArray = new Uint16Array (texArraySize);

        ramStats.sizeDataTextureIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(indices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB16UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGB_INTEGER,
            gl.UNSIGNED_SHORT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: document
     */
    generateTextureFor32BitIndices (gl, indices) {
        if (indices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }

        const textureWidth = 512;
        const textureHeight = Math.ceil (indices.length / 3 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 3;
        const texArray = new Uint32Array (texArraySize);

        ramStats.sizeDataTextureIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(indices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB32UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGB_INTEGER,
            gl.UNSIGNED_INT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: comment
     */
    generateTextureFor8BitsEdgeIndices (gl, edgeIndices) {
        if (edgeIndices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }

        const textureWidth = 512;
        const textureHeight = Math.ceil (edgeIndices.length / 2 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 2;
        const texArray = new Uint8Array (texArraySize);

        ramStats.sizeDataTextureEdgeIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(edgeIndices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG8UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RG_INTEGER,
            gl.UNSIGNED_BYTE,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: comment
     */
    generateTextureFor16BitsEdgeIndices (gl, edgeIndices) {
        if (edgeIndices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }

        const textureWidth = 512;
        const textureHeight = Math.ceil (edgeIndices.length / 2 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 2;
        const texArray = new Uint16Array (texArraySize);

        ramStats.sizeDataTextureEdgeIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(edgeIndices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG16UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RG_INTEGER,
            gl.UNSIGNED_SHORT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * TODO: comment
     */
    generateTextureFor32BitsEdgeIndices (gl, edgeIndices) {
        if (edgeIndices.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }

        const textureWidth = 512;
        const textureHeight = Math.ceil (edgeIndices.length / 2 / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 2;
        const texArray = new Uint32Array (texArraySize);

        ramStats.sizeDataTextureEdgeIndices +=texArray.byteLength;

        texArray.fill(0);
        texArray.set(edgeIndices, 0)

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RG32UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RG_INTEGER,
            gl.UNSIGNED_INT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {*} gl WebGL2Context 
     * @param {*} positions Array of (uniquified) positions in the layer
     * 
     * This will generate a texture for positions in the layer.
     * 
     * The texture will have:
     * - 512 columns, where each pixel will be a 16-bit-per-component RGB texture, corresponding to the XYZ of the position 
     * - a number of rows R where R*512 is just >= than the number of vertices (positions / 3)
     * 
     * @returns The created texture and its height
     */
    generateTextureForPositions (gl, positions) {
        const numVertices = positions.length / 3;
        const textureWidth = 512;
        const textureHeight =  Math.ceil (numVertices / textureWidth);

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight * 3;
        const texArray = new Uint16Array (texArraySize);

        ramStats.sizeDataTexturePositions +=texArray.byteLength;

        texArray.fill(0);

        texArray.set (positions, 0);

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB16UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RGB_INTEGER,
            gl.UNSIGNED_SHORT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     */
     generateTextureForPackedPortionIds (gl, portionIdsArray) {
        if (portionIdsArray.length == 0) {
            return {
                texture: null,
                textureHeight: 0,
            };
        }
        const lenArray = portionIdsArray.length;
        const textureWidth = 512;
        const textureHeight = Math.ceil (
            lenArray /
            textureWidth
        );

        if (textureHeight == 0)
        {
            throw "texture height == 0";
        }

        const texArraySize = textureWidth * textureHeight;
        const texArray = new Uint16Array (texArraySize);

        texArray.set (
            portionIdsArray,
            0
        );

        ramStats.sizeDataTexturePortionIds += texArray.byteLength;

        const texture = gl.createTexture();

        gl.bindTexture (gl.TEXTURE_2D, texture);

        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16UI, textureWidth, textureHeight);

        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            textureWidth,
            textureHeight,
            gl.RED_INTEGER,
            gl.UNSIGNED_SHORT,
            texArray,
            0
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return this.generateBindableTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
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