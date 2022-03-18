import {WEBGL_INFO} from "../../../../webglInfo.js";
import {ENTITY_FLAGS} from '../../ENTITY_FLAGS.js';
import {RENDER_PASSES} from '../../RENDER_PASSES.js';

import {math} from "../../../../math/math.js";
import {buildEdgeIndices} from '../../../../math/buildEdgeIndices.js';
import {RenderState} from "../../../../webgl/RenderState.js";
import {ArrayBuf} from "../../../../webgl/ArrayBuf.js";
import {geometryCompressionUtils} from "../../../../math/geometryCompressionUtils.js";
import {getBatchingRenderers} from "../trianglesBatching/TrianglesBatchingRenderers.js";
import {octEncodeNormals, quantizePositions} from "../../compression.js";
import * as uniquifyPositions from "../trianglesBatching/calculateUniquePositions.js";
import {
    ramStats,
    DataTextureState,
    DataTextureBuffer,
    DataTextureGenerator,
 } from "../DataTextureState.js"

const bigIndicesSupported = WEBGL_INFO.SUPPORTED_EXTENSIONS["OES_element_index_uint"];

const INDICES_EDGE_INDICES_ALIGNEMENT_SIZE = 8;

const tempUint8Vec4 = new Uint8Array(4);
const tempVec4a = math.vec4([0, 0, 0, 1]);
const tempVec4b = math.vec4([0, 0, 0, 1]);
const tempVec4c = math.vec4([0, 0, 0, 1]);
const tempVec3fa = new Float32Array(3);

const tempUint8Array4 = new Uint8Array (4);

const tempVec3a = math.vec3();
const tempVec3b = math.vec3();
const tempVec3c = math.vec3();
const tempVec3d = math.vec3();
const tempVec3e = math.vec3();
const tempVec3f = math.vec3();
const tempVec3g = math.vec3();

class DataPerGeometry
{
    /**
     * @param {int} vertexBase 
     * @param {ArrayLike<int>} positionsDecodeMatrix 
     * @param {ArrayLike<int>} obb 
     */
    constructor(vertexBase, positionsDecodeMatrix, obb)
    {
        /**
         * @type {int}
         */
        this.vertexBase = vertexBase;

        /**
         * @type {ArrayLike<int>}
         */

        this.positionsDecodeMatrix = positionsDecodeMatrix;

        /**
         * @type {ArrayLike<int>}
         */

        this.obb = obb;
    }
}

/**
 * @private
 */
class TrianglesInstancingLayer {

    /**
     * @param model
     * @param cfg
     * @param cfg.layerIndex
     * @param cfg.positions Flat float Local-space positions array.
     * @param [cfg.normals] Flat float normals array.
     * @param cfg.indices Flat int indices array.
     * @param [cfg.edgeIndices] Flat int edges indices array.
     * @param cfg.edgeThreshold
     * @param cfg.origin
     * @params cfg.solid
     */
    constructor(model, cfg) {

        console.log ("create instancing layer");

        /**
         * State sorting key.
         * @type {string}
         */
        this.sortId = "TrianglesInstancingLayer";

        /**
         * Index of this InstancingLayer in PerformanceModel#_layerList
         * @type {Number}
         */
        this.layerIndex = cfg.layerIndex;

        this._batchingRenderers = getBatchingRenderers(model.scene);

        /**
         * @type PeformanceModel
         */
        this.model = model;

        this._aabb = math.collapseAABB3();

        const stateCfg = {
            positionsDecodeMatrix: math.mat4(),
            numInstances: 0,
            obb: math.OBB3(),
            origin: null
        };
         
        /**
         * Registered geometrical data that portions will use.
         * 
         * - `key` is the geometry id
         * - `value` is the geometry data
         * 
         * @private
         */
        this._registerdGeometries = {};

        /**
         * Tells if a given geometry is already loaded into internal data arrays.
         * 
         * - `key` is the geometry id
         * - `value` is true if the geometry is already loaded
         * 
         * @private
         */
        this._alreadyLoadedGeometries = {};

        /**
         * @type RenderState
         */
        this._state = new RenderState(stateCfg);

        /**
         * @type DataTextureState
         */
        this._dataTextureState = new DataTextureState();

        this._state.textureState = this._dataTextureState;

        /**
         * @type {DataTextureGenerator}
         */
        this.dataTextureGenerator = new DataTextureGenerator();

        if (!this.model.cameraTexture)
        {
            this.model.cameraTexture = this.dataTextureGenerator.generateCameraDataTexture (
                this.model.scene.canvas.gl,
                this.model.scene.camera,
                this.model.scene,
                [ 0, 0, 0 ]
            );
        }

        this._dataTextureState.textureCameraMatrices = this.model.cameraTexture;

        // These counts are used to avoid unnecessary render passes
        this._numPortions = 0;
        this._numVisibleLayerPortions = 0;
        this._numTransparentLayerPortions = 0;
        this._numXRayedLayerPortions = 0;
        this._numHighlightedLayerPortions = 0;
        this._numSelectedLayerPortions = 0;
        this._numClippableLayerPortions = 0;
        this._numEdgesLayerPortions = 0;
        this._numPickableLayerPortions = 0;
        this._numCulledLayerPortions = 0;
                
        this._finalized = false;

        /**
         * The axis-aligned World-space boundary of this InstancingLayer's positions.
         * @type {*|Float64Array}
         */
        this.aabb = math.collapseAABB3();
        
        /**
         * Per-geometry data that needs to be taken into account when instancing it.
         * 
         * @type {Map<string, DataPerGeometry}
         */
        this._dataPerGeometry = {};

        this._buffer = new DataTextureBuffer();

        // this._buffer = {
        //     positions: [],
        //     indices8Bits: [],
        //     indices16Bits: [],
        //     indices32Bits: [],
        //     edgeIndices8Bits: [],
        //     edgeIndices16Bits: [],
        //     edgeIndices32Bits: [],
        //     edgeIndices: [],
        //     _objectDataColors: [],
        //     _objectDataPickColors: [],
        //     _vertexBasesForObject: [],
        //     _objectDataPositionsMatrices: [],
        //     _objectDataInstanceGeometryMatrices: [],
        //     _objectDataInstanceNormalsMatrices: [],
        //     _portionIdForIndices8Bits: [],
        //     _portionIdForIndices16Bits: [],
        //     _portionIdForIndices32Bits: [],
        //     _portionIdForEdges8Bits: [],
        //     _portionIdForEdges16Bits: [],
        //     _portionIdForEdges32Bits: [],
        //     _portionIdFanOut: [],
        // };
    }

    loadGeometryData (cfg)
    {
        const geometryId = cfg.id;

        const preCompressed = (!!cfg.positionsDecodeMatrix);
        const pickSurfacePrecisionEnabled = this.model.scene.pickSurfacePrecisionEnabled;
        const gl = this.model.scene.canvas.gl;

        const buffer = this._buffer;

        let positionsDecodeMatrix = cfg.positionsDecodeMatrix;

        // Quantize positions if needed
        if (!positionsDecodeMatrix)
        {
            positionsDecodeMatrix = math.identityMat4();

            const localAABB = math.collapseAABB3();
            try {
            math.expandAABB3Points3(localAABB, cfg.positions);
            }catch (e) {
                debugger;
            }
            math.AABB3ToOBB3(localAABB);

            cfg.positions = quantizePositions(cfg.positions, localAABB, stateCfg.positionsDecodeMatrix);
        }

        let edgeIndices = cfg.edgeIndices;

        // Calculate Edge Indices if needed
        if (!edgeIndices) {
            edgeIndices = buildEdgeIndices(cfg.positions, cfg.indices, null, cfg.edgeThreshold || 10);
        }

        // Unique-ify positions
        let uniquePositions, uniqueIndices, uniqueEdgeIndices;

        [
            uniquePositions,
            uniqueIndices,
            uniqueEdgeIndices,
        ] = uniquifyPositions.uniquifyPositions ({
            positions: cfg.positions,
            indices: cfg.indices,
            edgeIndices: edgeIndices
        });

        // Indices alignement
        {
            const alignedIndicesLen = Math.ceil ((uniqueIndices.length / 3) / INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) * INDICES_EDGE_INDICES_ALIGNEMENT_SIZE * 3;

            // ramStats.overheadSizeAlignementIndices += 2 * (alignedIndicesLen - uniqueIndices.length);

            {
                const alignedIndices = new Uint32Array(alignedIndicesLen);
                alignedIndices.fill(0);
                alignedIndices.set (uniqueIndices);
                uniqueIndices = alignedIndices;
            }
        }
        
        // EdgeIndices alignement
        {
            const alignedEdgeIndicesLen = Math.ceil ((uniqueEdgeIndices.length / 2) / INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) * INDICES_EDGE_INDICES_ALIGNEMENT_SIZE * 2;

            // ramStats.overheadSizeAlignementEdgeIndices += 2 * (alignedEdgeIndicesLen - uniqueEdgeIndices.length);

            {
                const alignedEdgeIndices = new Uint32Array(alignedEdgeIndicesLen);
                alignedEdgeIndices.fill(0);
                alignedEdgeIndices.set (uniqueEdgeIndices);
                uniqueEdgeIndices = alignedEdgeIndices;
            }
        }

        // // TODO: per-geometry?
        // this._positionDecodeMatricesForGeometry [geometryId] = positionsDecodeMatrix;
        // // buffer._objectDataPositionsMatrices.push (positionsDecodeMatrix);

        // Calculate geomrtry AABB
        const localAABB = math.collapseAABB3();
        math.expandAABB3Points3(localAABB, uniquePositions);
        geometryCompressionUtils.decompressAABB(localAABB, positionsDecodeMatrix);
        const geometryOBB = math.AABB3ToOBB3(localAABB);

        this._dataPerGeometry[geometryId] = new DataPerGeometry (
            buffer.positions.length / 3,
            positionsDecodeMatrix,
            geometryOBB
        );

        // this._vertexBasesForGeometry[geometryId] = buffer.positions.length / 3;

        // Buffer positions
        for (let i = 0, len = uniquePositions.length; i < len; i++) {
            buffer.positions.push(uniquePositions[i]);
        }

        const numUniquePositions = uniquePositions.length / 3;

        // Buffer indices
        {
            let triangleNumber = 0;
            
            for (let i = 0, len = uniqueIndices.length; i < len; i+=3) {
                if (numUniquePositions <= (1<< 8)) {
                    buffer.indices8Bits.push(uniqueIndices[i]);
                    buffer.indices8Bits.push(uniqueIndices[i+1]);
                    buffer.indices8Bits.push(uniqueIndices[i+2]);
                } else if (numUniquePositions <= (1<< 16)) {
                    buffer.indices16Bits.push(uniqueIndices[i]);
                    buffer.indices16Bits.push(uniqueIndices[i+1]);
                    buffer.indices16Bits.push(uniqueIndices[i+2]);
                } else {
                    buffer.indices32Bits.push(uniqueIndices[i]);
                    buffer.indices32Bits.push(uniqueIndices[i+1]);
                    buffer.indices32Bits.push(uniqueIndices[i+2]);
                }

                if ((triangleNumber % INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) == 0) {
                    if (numUniquePositions <= (1<< 8)) {
                        // TODO: per-geometry?
                        buffer._portionIdForIndices8Bits.push (this._numPortions);
                    } else if (numUniquePositions <= (1<< 16)) {
                        // TODO: per-geometry?
                        buffer._portionIdForIndices16Bits.push (this._numPortions);
                    } else {
                        // TODO: per-geometry?
                        buffer._portionIdForIndices32Bits.push (this._numPortions);
                    }
                }

                triangleNumber++;
            }
        }

        // Buffer edge indices
        {
            let edgeNumber = 0;

            for (let i = 0, len = uniqueEdgeIndices.length; i < len; i+=2) {
                if (numUniquePositions <= (1<< 8)) {
                    buffer.edgeIndices8Bits.push(uniqueEdgeIndices[i]);
                    buffer.edgeIndices8Bits.push(uniqueEdgeIndices[i+1]);
                } else if (numUniquePositions <= (1<< 16)) {
                    buffer.edgeIndices16Bits.push(uniqueEdgeIndices[i]);
                    buffer.edgeIndices16Bits.push(uniqueEdgeIndices[i+1]);
                } else {
                    buffer.edgeIndices32Bits.push(uniqueEdgeIndices[i]);
                    buffer.edgeIndices32Bits.push(uniqueEdgeIndices[i+1]);
                }

                if ((edgeNumber % INDICES_EDGE_INDICES_ALIGNEMENT_SIZE) == 0) {
                    if (numUniquePositions <= (1<< 8)) {
                        // TODO: per-geometry?
                        buffer._portionIdForEdges8Bits.push (this._numPortions);
                    } else if (numUniquePositions <= (1<< 16)) {
                        // TODO: per-geometry?
                        buffer._portionIdForEdges16Bits.push (this._numPortions);
                    } else {
                        // TODO: per-geometry?
                        buffer._portionIdForEdges32Bits.push (this._numPortions);
                    }
                }

                edgeNumber++;
            }
        }

        // TODO
        /** @private */
        this.numIndices = (cfg.indices) ? cfg.indices.length / 3 : 0;

        // Vertex arrays
        // this._metallicRoughness = [];
        // this._offsets = [];

        // // Modeling matrix per instance, array for each column
        // this._modelMatrixCol0 = [];
        // this._modelMatrixCol1 = [];
        // this._modelMatrixCol2 = [];

        // // Modeling normal matrix per instance, array for each column
        // this._modelNormalMatrixCol0 = [];
        // this._modelNormalMatrixCol1 = [];
        // this._modelNormalMatrixCol2 = [];

        this._portions = [];

        if (cfg.origin) {
            this._state.origin = math.vec3(cfg.origin);
        }

        /**
         * When true, this layer contains solid triangle meshes, otherwise this layer contains surface triangle meshes
         * @type {boolean}
         */
        this.solid = !!cfg.solid;
    }

    /**
     * Creates a new portion within this InstancingLayer, returns the new portion ID.
     *
     * The portion will instance this InstancingLayer's geometry.
     *
     * Gives the portion the specified color and matrix.
     *
     * @param cfg Portion params
     * @param cfg.color Color [0..255,0..255,0..255]
     * @param cfg.metallic Metalness factor [0..255]
     * @param cfg.roughness Roughness factor [0..255]
     * @param cfg.opacity Opacity [0..255].
     * @param cfg.meshMatrix Flat float 4x4 matrix.
     * @param [cfg.worldMatrix] Flat float 4x4 matrix.
     * @param cfg.worldAABB Flat float AABB.
     * @param cfg.pickColor Quantized pick color
     * @returns {number} Portion ID.
     */
    createPortion(cfg) {
        const geometryId = cfg.geometryId;

        if (!this._alreadyLoadedGeometries [geometryId]) {
            this.loadGeometryData (
                this.getGeometryData (geometryId)
            );
        }

        const color = cfg.color;
        const metallic = cfg.metallic;
        const roughness = cfg.roughness;
        const opacity = cfg.opacity;
        const meshMatrix = cfg.meshMatrix;
        const worldMatrix = cfg.worldMatrix;
        const worldAABB = cfg.aabb;
        const pickColor = cfg.pickColor;

        if (this._finalized) {
            throw "Already finalized";
        }

        const buffer = this._buffer;

        /**
         * @type {DataPerGeometry}
         */
        const dataPerGeometry = this._dataPerGeometry[geometryId];

        buffer._vertexBasesForObject.push (
            dataPerGeometry.vertexBase
        );

        buffer._objectDataPositionsMatrices.push (
            dataPerGeometry.positionsDecodeMatrix
        );

        buffer._objectDataColors.push ([
            color[0], // Color is pre-quantized by PerformanceModel,
            color[1],
            color[2],
            opacity,
        ])

        buffer._objectDataPickColors.push (pickColor);

        // TODO: find AABB for portion by transforming the geometry local AABB by the given meshMatrix?

        // const r = color[0]; // Color is pre-quantized by PerformanceModel,
        // const g = color[1];
        // const b = color[2];
        // const a = color[3];

        // this._colors.push(r);
        // this._colors.push(g);
        // this._colors.push(b);
        // this._colors.push(opacity);

        // this._metallicRoughness.push((metallic !== null && metallic !== undefined) ? metallic : 0);
        // this._metallicRoughness.push((roughness !== null && roughness !== undefined) ? roughness : 255);

        // if (this.model.scene.entityOffsetsEnabled) {
        //     this._offsets.push(0);
        //     this._offsets.push(0);
        //     this._offsets.push(0);
        // }

        // this._modelMatrixCol0.push(meshMatrix[0]);
        // this._modelMatrixCol0.push(meshMatrix[4]);
        // this._modelMatrixCol0.push(meshMatrix[8]);
        // this._modelMatrixCol0.push(meshMatrix[12]);

        // this._modelMatrixCol1.push(meshMatrix[1]);
        // this._modelMatrixCol1.push(meshMatrix[5]);
        // this._modelMatrixCol1.push(meshMatrix[9]);
        // this._modelMatrixCol1.push(meshMatrix[13]);

        // this._modelMatrixCol2.push(meshMatrix[2]);
        // this._modelMatrixCol2.push(meshMatrix[6]);
        // this._modelMatrixCol2.push(meshMatrix[10]);
        // this._modelMatrixCol2.push(meshMatrix[14]);

        // Mesh instance matrix
        buffer._objectDataInstanceGeometryMatrices.push (
            meshMatrix
        );

        // if (this._state.normalsBuf) {

        //     // Note: order of inverse and transpose doesn't matter

        //     let transposedMat = math.transposeMat4(meshMatrix, math.mat4()); // TODO: Use cached matrix
        //     let normalMatrix = math.inverseMat4(transposedMat);

        //     this._modelNormalMatrixCol0.push(normalMatrix[0]);
        //     this._modelNormalMatrixCol0.push(normalMatrix[4]);
        //     this._modelNormalMatrixCol0.push(normalMatrix[8]);
        //     this._modelNormalMatrixCol0.push(normalMatrix[12]);

        //     this._modelNormalMatrixCol1.push(normalMatrix[1]);
        //     this._modelNormalMatrixCol1.push(normalMatrix[5]);
        //     this._modelNormalMatrixCol1.push(normalMatrix[9]);
        //     this._modelNormalMatrixCol1.push(normalMatrix[13]);

        //     this._modelNormalMatrixCol2.push(normalMatrix[2]);
        //     this._modelNormalMatrixCol2.push(normalMatrix[6]);
        //     this._modelNormalMatrixCol2.push(normalMatrix[10]);
        //     this._modelNormalMatrixCol2.push(normalMatrix[14]);
        // }

        // Mesh instance normal matrix
        {
            // Note: order of inverse and transpose doesn't matter
            let transposedMat = math.transposeMat4(meshMatrix, math.mat4()); // TODO: Use cached matrix
            let normalMatrix = math.inverseMat4(transposedMat);

            buffer._objectDataInstanceNormalsMatrices.push (
                normalMatrix
            );
        }

        // // Per-vertex pick colors

        // this._pickColors.push(pickColor[0]);
        // this._pickColors.push(pickColor[1]);
        // this._pickColors.push(pickColor[2]);
        // this._pickColors.push(pickColor[3]);

        // Expand AABB
        math.collapseAABB3(worldAABB);

        const obb = dataPerGeometry.obb;

        for (let i = 0, len = obb.length;  i < len; i += 4) {
            tempVec4a[0] = obb[i + 0];
            tempVec4a[1] = obb[i + 1];
            tempVec4a[2] = obb[i + 2];
            math.transformPoint4(meshMatrix, tempVec4a, tempVec4b);
            if (worldMatrix) {
                math.transformPoint4(worldMatrix, tempVec4b, tempVec4c);
                math.expandAABB3Point3(worldAABB, tempVec4c);
            } else {
                math.expandAABB3Point3(worldAABB, tempVec4b);
            }
        }

        // if (this._state.origin) {
        //     const origin = this._state.origin;
        //     worldAABB[0] += origin[0];
        //     worldAABB[1] += origin[1];
        //     worldAABB[2] += origin[2];
        //     worldAABB[3] += origin[0];
        //     worldAABB[4] += origin[1];
        //     worldAABB[5] += origin[2];
        // }

        math.expandAABB3(this.aabb, worldAABB);

        this._state.numInstances++;

        const portionId = this._portions.length;

        const portion = {};

        // if (this.model.scene.pickSurfacePrecisionEnabled) {
        //     portion.matrix = meshMatrix.slice();
        //     portion.inverseMatrix = null; // Lazy-computed in precisionRayPickSurface
        //     portion.normalMatrix = null; // Lazy-computed in precisionRayPickSurface
        // }

        this._portions.push(portion);

        this._numPortions++;
        this.model.numPortions++;

        return portionId;
    }

    finalize() {
        if (this._finalized) {
            throw "Already finalized";
        }

        const gl = this.model.scene.canvas.gl;
        const buffer = this._buffer;
        const textureState = this._dataTextureState;
        const state = this._state;

        // a) colors and flags texture
        textureState.texturePerObjectIdColorsAndFlags = this.dataTextureGenerator.generateTextureForColorsAndFlags (
            gl,
            buffer._objectDataColors,
            buffer._objectDataPickColors,
            buffer._vertexBasesForObject
        );

        // b) positions decode matrices texture
        textureState.texturePerObjectIdPositionsDecodeMatrix = this.dataTextureGenerator.generateTextureForPositionsDecodeMatrices (
            gl,
            buffer._objectDataPositionsMatrices,
            buffer._objectDataInstanceGeometryMatrices,
            buffer._objectDataInstanceNormalsMatrices
        );

        // c) position coordinates texture
        textureState.texturePerVertexIdCoordinates = this.dataTextureGenerator.generateTextureForPositions (
            gl,
            buffer.positions
        );

        // d) portion Id triangles texture
        textureState.texturePerPolygonIdPortionIds8Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForIndices8Bits
        );

        textureState.texturePerPolygonIdPortionIds16Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForIndices16Bits
        );

        textureState.texturePerPolygonIdPortionIds32Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForIndices32Bits
        );

        // e) portion Id texture for edges
        textureState.texturePerEdgeIdPortionIds8Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForEdges8Bits
        );

        textureState.texturePerEdgeIdPortionIds16Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForEdges16Bits
        );

        textureState.texturePerEdgeIdPortionIds32Bits = this.dataTextureGenerator.generateTextureForPackedPortionIds (
            gl,
            buffer._portionIdForEdges32Bits
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

        this._buffer = null;

        // if (this._metallicRoughness.length > 0) {
        //     const metallicRoughness = new Uint8Array(this._metallicRoughness);
        //     let normalized = false;
        //     this._state.metallicRoughnessBuf = new ArrayBuf(gl, gl.ARRAY_BUFFER, metallicRoughness, this._metallicRoughness.length, 2, gl.STATIC_DRAW, normalized);
        // }
        // if (this.model.scene.entityOffsetsEnabled) {
        //     if (this._offsets.length > 0) {
        //         const notNormalized = false;
        //         this._state.offsetsBuf = new ArrayBuf(gl, gl.ARRAY_BUFFER, new Float32Array(this._offsets), this._offsets.length, 3, gl.DYNAMIC_DRAW, notNormalized);
        //         this._offsets = []; // Release memory
        //     }
        // }

        this._finalized = true;
    }

    // The following setters are called by PerformanceMesh, in turn called by PerformanceNode, only after the layer is finalized.
    // It's important that these are called after finalize() in order to maintain integrity of counts like _numVisibleLayerPortions etc.

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
        this._setFlags(portionId, flags, meshTransparent);
        this._setFlags2(portionId, flags);
    }

    setVisible(portionId, flags, meshTransparent) {
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
        this._setFlags(portionId, flags, meshTransparent);
    }

    setHighlighted(portionId, flags, meshTransparent) {
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
        this._setFlags(portionId, flags, meshTransparent);
    }

    setXRayed(portionId, flags, meshTransparent) {
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
        this._setFlags(portionId, flags, meshTransparent);
    }

    setSelected(portionId, flags, meshTransparent) {
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
        this._setFlags(portionId, flags, meshTransparent);
    }

    setEdges(portionId, flags, meshTransparent) {
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
        this._setFlags(portionId, flags, meshTransparent);
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

    setCollidable(portionId, flags) {
        if (!this._finalized) {
            throw "Not finalized";
        }
    }

    setPickable(portionId, flags, meshTransparent) {
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
        this._setFlags2(portionId, flags, meshTransparent);
    }

    setCulled(portionId, flags, meshTransparent) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (flags & ENTITY_FLAGS.CULLED) {
            this._numCulledLayerPortions++;
            this.model.numCulledLayerPortions++;
        } else {
            this._numCulledLayerPortions--;
            this.model.numCulledLayerPortions--;
        }
        this._setFlags(portionId, flags, meshTransparent);
    }

    setColor(portionId, color) { // RGBA color is normalized as ints
        if (!this._finalized) {
            throw "Not finalized";
        }
        tempUint8Vec4[0] = color[0];
        tempUint8Vec4[1] = color[1];
        tempUint8Vec4[2] = color[2];
        tempUint8Vec4[3] = color[3];
        if (this._state.colorsBuf) {
            this._state.colorsBuf.setData(tempUint8Vec4, portionId * 4, 4);
        }
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

    // setMatrix(portionId, matrix) {
    //
    //     if (!this._finalized) {
    //         throw "Not finalized";
    //     }
    //
    //     var offset = portionId * 4;
    //
    //     tempFloat32Vec4[0] = matrix[0];
    //     tempFloat32Vec4[1] = matrix[4];
    //     tempFloat32Vec4[2] = matrix[8];
    //     tempFloat32Vec4[3] = matrix[12];
    //
    //     this._state.modelMatrixCol0Buf.setData(tempFloat32Vec4, offset, 4);
    //
    //     tempFloat32Vec4[0] = matrix[1];
    //     tempFloat32Vec4[1] = matrix[5];
    //     tempFloat32Vec4[2] = matrix[9];
    //     tempFloat32Vec4[3] = matrix[13];
    //
    //     this._state.modelMatrixCol1Buf.setData(tempFloat32Vec4, offset, 4);
    //
    //     tempFloat32Vec4[0] = matrix[2];
    //     tempFloat32Vec4[1] = matrix[6];
    //     tempFloat32Vec4[2] = matrix[10];
    //     tempFloat32Vec4[3] = matrix[14];
    //
    //     this._state.modelMatrixCol2Buf.setData(tempFloat32Vec4, offset, 4);
    // }

    _setFlags(portionId, flags, meshTransparent) {

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

        // Normal fill

        let f0;
        if (!visible || culled || xrayed) { // Highlight & select are layered on top of color - not mutually exclusive
            f0 = RENDER_PASSES.NOT_RENDERED;
        } else {
            if (meshTransparent) {
                f0 = RENDER_PASSES.COLOR_TRANSPARENT;
            } else {
                f0 = RENDER_PASSES.COLOR_OPAQUE;
            }
        }

        // Emphasis fill

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
            if (meshTransparent) {
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

    _setFlags2(portionId, flags) {

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

    setOffset(portionId, offset) {
        if (!this._finalized) {
            throw "Not finalized";
        }
        if (!this.model.scene.entityOffsetsEnabled) {
            this.model.error("Entity#offset not enabled for this Viewer"); // See Viewer entityOffsetsEnabled
            return;
        }
        tempVec3fa[0] = offset[0];
        tempVec3fa[1] = offset[1];
        tempVec3fa[2] = offset[2];
        if (this._state.offsetsBuf) {
            this._state.offsetsBuf.setData(tempVec3fa, portionId * 3, 3);
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
            // Only opaque, filled objects can be occluders
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

    //-----------------------------------------------------------------------------------------

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

        if (!portion.inverseMatrix) {
            portion.inverseMatrix = math.inverseMat4(portion.matrix, math.mat4());
        }

        if (worldNormal && !portion.normalMatrix) {
            portion.normalMatrix = math.transposeMat4(portion.inverseMatrix, math.mat4());
        }

        const quantizedPositions = state.quantizedPositions;
        const indices = state.indices;
        const origin = state.origin;
        const offset = portion.offset;

        const rtcRayOrigin = tempVec3a;
        const rtcRayDir = tempVec3b;

        rtcRayOrigin.set(origin ? math.subVec3(worldRayOrigin, origin, tempVec3c) : worldRayOrigin);  // World -> RTC
        rtcRayDir.set(worldRayDir);

        if (offset) {
            math.subVec3(rtcRayOrigin, offset);
        }

        math.transformRay(this.model.worldNormalMatrix, rtcRayOrigin, rtcRayDir, rtcRayOrigin, rtcRayDir);

        math.transformRay(portion.inverseMatrix, rtcRayOrigin, rtcRayDir, rtcRayOrigin, rtcRayDir);

        const a = tempVec3d;
        const b = tempVec3e;
        const c = tempVec3f;

        let gotIntersect = false;
        let closestDist = 0;
        const closestIntersectPos = tempVec3g;

        for (let i = 0, len = indices.length; i < len; i += 3) {

            const ia = indices[i + 0] * 3;
            const ib = indices[i + 1] * 3;
            const ic = indices[i + 2] * 3;

            a[0] = quantizedPositions[ia];
            a[1] = quantizedPositions[ia + 1];
            a[2] = quantizedPositions[ia + 2];

            b[0] = quantizedPositions[ib];
            b[1] = quantizedPositions[ib + 1];
            b[2] = quantizedPositions[ib + 2];

            c[0] = quantizedPositions[ic];
            c[1] = quantizedPositions[ic + 1];
            c[2] = quantizedPositions[ic + 2];

            math.decompressPosition(a, state.positionsDecodeMatrix);
            math.decompressPosition(b, state.positionsDecodeMatrix);
            math.decompressPosition(c, state.positionsDecodeMatrix);

            if (math.rayTriangleIntersect(rtcRayOrigin, rtcRayDir, a, b, c, closestIntersectPos)) {

                math.transformPoint3(portion.matrix, closestIntersectPos, closestIntersectPos);

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
            math.transformVec3(portion.normalMatrix, worldNormal, worldNormal);
            math.transformVec3(this.model.worldNormalMatrix, worldNormal, worldNormal);
            math.normalizeVec3(worldNormal);
        }

        return gotIntersect;
    }

    destroy() {
        const state = this._state;
        if (state.positionsBuf) {
            state.positionsBuf.destroy();
            state.positionsBuf = null;
        }
        if (state.normalsBuf) {
            state.normalsBuf.destroy();
            state.normalsBuf = null;
        }
        if (state.colorsBuf) {
            state.colorsBuf.destroy();
            state.colorsBuf = null;
        }
        if (state.metallicRoughnessBuf) {
            state.metallicRoughnessBuf.destroy();
            state.metallicRoughnessBuf = null;
        }
        if (state.flagsBuf) {
            state.flagsBuf.destroy();
            state.flagsBuf = null;
        }
        if (state.flags2Buf) {
            state.flags2Buf.destroy();
            state.flags2Buf = null;
        }
        if (state.offsetsBuf) {
            state.offsetsBuf.destroy();
            state.offsetsBuf = null;
        }
        if (state.modelMatrixCol0Buf) {
            state.modelMatrixCol0Buf.destroy();
            state.modelMatrixCol0Buf = null;
        }
        if (state.modelMatrixCol1Buf) {
            state.modelMatrixCol1Buf.destroy();
            state.modelMatrixCol1Buf = null;
        }
        if (state.modelMatrixCol2Buf) {
            state.modelMatrixCol2Buf.destroy();
            state.modelMatrixCol2Buf = null;
        }
        if (state.modelNormalMatrixCol0Buf) {
            state.modelNormalMatrixCol0Buf.destroy();
            state.modelNormalMatrixCol0Buf = null;
        }
        if (state.modelNormalMatrixCol1Buf) {
            state.modelNormalMatrixCol1Buf.destroy();
            state.modelNormalMatrixCol1Buf = null;
        }
        if (state.modelNormalMatrixCol2Buf) {
            state.modelNormalMatrixCol2Buf.destroy();
            state.modelNormalMatrixCol2Buf = null;
        }
        if (state.indicesBuf) {
            state.indicesBuf.destroy();
            state.indicessBuf = null;
        }
        if (state.edgeIndicesBuf) {
            state.edgeIndicesBuf.destroy();
            state.edgeIndicessBuf = null;
        }
        if (state.pickColorsBuf) {
            state.pickColorsBuf.destroy();
            state.pickColorsBuf = null;
        }
        state.destroy();
    }

    hasGeometry (geometryId)
    {
        return (geometryId in this._registerdGeometries);
    }

    registerGeometry (cfg)
    {
        this._registerdGeometries [cfg.id] = cfg;
    }

    getGeometryData (geometryId)
    {
        return this._registerdGeometries[geometryId];    
    }

    getGeometryOrigin (geometryId)
    {
        return this._registerdGeometries[geometryId].origin;
    }
}

export {TrianglesInstancingLayer};