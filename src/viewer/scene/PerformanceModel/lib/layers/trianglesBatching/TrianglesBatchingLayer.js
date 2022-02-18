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

// 12-bits allowed for object ids
const MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER = (1 << 12);

// 2048 is max data texture height
const MAX_DATA_TEXTURE_HEIGHT = (1 << 11);

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
    cannotCreatePortion: {
        because10BitsObjectId: 0,
        becauseTextureSize: 0,
    }
};

let _numTotalPolygons = 0;
let _numTotalEdges = 0;
let _numTotalEdges2 = 0;
let _numTotalVertices = 0;
let _numUniqueSmallVertices = 0;

let _lastCanCreatePortion = {
    positions: null,
    indices: null,
    edgeIndices: null,
    uniquePositions: null,
    uniqueIndices: null,
    uniqueEdgeIndices: null,
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

        this._state = new RenderState({
            positionsBuf: null,
            offsetsBuf: null,
            normalsBuf: null,
            colorsBuf: null,
            metallicRoughnessBuf: null,
            flagsBuf: null,
            flags2Buf: null,
            indicesBuf: null,
            edgeIndicesBuf: null,
            positionsDecodeMatrix: math.mat4(),
            texturePerObjectIdPositionsDecodeMatrix: null,
            texturePerObjectIdPositionsDecodeMatrixHeight: null,
            texturePerVertexIdCoordinates: null,
            texturePerVertexIdCoordinatesHeight: null,
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

        this._numIndicesInLayer = 0;
        this._numEdgeIndicesInLayer = 0;

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

        this._portionIdForIndices = []; // chipmunk
        this._portionIdForEdges = []; // chipmunk

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
        
        _lastCanCreatePortion.positions = positions;
        _lastCanCreatePortion.indices = indices;
        _lastCanCreatePortion.edgeIndices = edgeIndices;

        [
            _lastCanCreatePortion.uniquePositions,
            _lastCanCreatePortion.uniqueIndices,
            _lastCanCreatePortion.uniqueEdgeIndices,
        ] = uniquifyPositions.uniquifyPositions ({
            positions,
            indices,
            edgeIndices
        });

        _numUniqueSmallVertices = _numUniqueSmallVertices + _lastCanCreatePortion.uniquePositions.length;
        _numTotalVertices += positions.length;
        _numTotalPolygons += indices.length / 3;
        _numTotalEdges += edgeIndices.length / 2;

        if (!(this._numPortions < MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER))
        {
            ramStats.cannotCreatePortion.because10BitsObjectId++;
        }

        if (!(((this._numUniqueVerts + (_lastCanCreatePortion.uniquePositions.length / 3)) / 512) <= MAX_DATA_TEXTURE_HEIGHT &&
            ((this._numIndicesInLayer + (_lastCanCreatePortion.uniqueIndices.length / 3)) / 512) <= MAX_DATA_TEXTURE_HEIGHT))
        {
            ramStats.cannotCreatePortion.becauseTextureSize++;
        }

        let retVal = this._numPortions < MAX_NUMBER_OBJECTS_IN_BATCHING_LAYER && 
                     ((this._numUniqueVerts + (_lastCanCreatePortion.uniquePositions.length / 3)) / 512) <= MAX_DATA_TEXTURE_HEIGHT &&
                     ((this._numIndicesInLayer + (_lastCanCreatePortion.uniqueIndices.length / 3)) / 512) <= MAX_DATA_TEXTURE_HEIGHT;

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

        ramStats.numberOfPortions++;

        cfg.positions = _lastCanCreatePortion.uniquePositions || [];
        cfg.indices = _lastCanCreatePortion.uniqueIndices;
        cfg.edgeIndices = _lastCanCreatePortion.uniqueEdgeIndices;

        if ((cfg.positions.length / 3) > (1<<16))
        {
            console.log (`YAY! ${(cfg.positions.length / 3)} positions`);
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

        if (normals && normals.length > 0) {

            if (this._preCompressed) {

                for (let i = 0, len = normals.length; i < len; i++) {
                    buffer.normals.push(normals[i]);
                }

            } else {

                const worldNormalMatrix = tempMat4;

                if (meshMatrix) {
                    math.inverseMat4(math.transposeMat4(meshMatrix, tempMat4b), worldNormalMatrix); // Note: order of inverse and transpose doesn't matter

                } else {
                    math.identityMat4(worldNormalMatrix, worldNormalMatrix);
                }

                transformAndOctEncodeNormals(worldNormalMatrix, normals, normals.length, buffer.normals, buffer.normals.length);
            }
        }

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

        if (indices) {
            for (let i = 0, len = indices.length; i < len; i+=3) {
                buffer.indices.push(indices[i]);
                buffer.indices.push(indices[i+1]);
                buffer.indices.push(indices[i+2]);
                this._portionIdForIndices.push (this._numPortions);
            }
            this._numIndicesInLayer += indices.length; // chupmunk
        }

        if (edgeIndices) {
            for (let i = 0, len = edgeIndices.length; i < len; i+=2) {
                buffer.edgeIndices.push(edgeIndices[i]);
                buffer.edgeIndices.push(edgeIndices[i+1]);
                this._portionIdForEdges.push (this._numPortions);
            }
            this._numEdgeIndicesInLayer += edgeIndices.length; // chupmunk
        }

        // start of chipmunk
        this._objectDataPickColors.push (
            pickColor
        );
        // end of chipmunk

        if (scene.entityOffsetsEnabled) {
            for (let i = 0; i < numVerts; i++) {
                buffer.offsets.push(0);
                buffer.offsets.push(0);
                buffer.offsets.push(0);
            }
        }

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
            positions: null,
            indices: null,
            edgeIndices: null,
            uniquePositions: null,
            uniqueIndices: null,
            uniqueEdgeIndices: null,
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
        const gl = this.model.scene.canvas.gl;
        const buffer = this._buffer;

        _numTotalEdges2 += buffer.edgeIndices.length / 2;

        // start of chipmunk
        // console.log (JSON.stringify({
        //     'total-vertices-so-far': _numTotalVertices,
        //     'unique-small-vertices-so-far': _numUniqueSmallVertices,
        //     'total-polygons': [
        //         _numTotalPolygons, buffer.indices.length / 3
        //     ],
        //     'total-edges': [
        //         _numTotalEdges, _numTotalEdges2, buffer.edgeIndices.length / 2, this._numEdgeIndicesInLayer / 2
        //     ],
        //     'ratio': (_numUniqueSmallVertices / _numTotalVertices * 100).toFixed(2)
        // }, null, 4));

        // Generate all the needed textures in the layer

        // a) colors and flags texture
        const colorsAndFlagsTexture = this.generateTextureForColorsAndFlags (
            gl,
            this._objectDataColors,
            this._objectDataPickColors,
            this._vertexBasesForObject
        );

        state.texturePerObjectIdColorsAndFlags = colorsAndFlagsTexture.texture;
        state.texturePerObjectIdColorsAndFlagsHeight = colorsAndFlagsTexture.textureHeight;

        // b) positions decode matrices texture
        const decodeMatrixTexture = this.generateTextureForPositionsDecodeMatrices (
            gl,
            this._objectDataPositionsMatrices
        ); 

        state.texturePerObjectIdPositionsDecodeMatrix = decodeMatrixTexture.texture;
        state.texturePerObjectIdPositionsDecodeMatrixHeight = decodeMatrixTexture.textureHeight;

        // c) position coordinates texture
        const texturePerVertexIdCoordinates = this.generateTextureForPositions (
            gl,
            buffer.positions
        );

        state.texturePerVertexIdCoordinates = texturePerVertexIdCoordinates.texture;
        state.texturePerVertexIdCoordinatesHeight = texturePerVertexIdCoordinates.textureHeight;

        // d) portion Id triangles texture
        const texturePerPolygonIdPortionIds = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForIndices
        );

        state.texturePerPolygonIdPortionIds = texturePerPolygonIdPortionIds.texture;
        state.texturePerPolygonIdPortionIdsHeight = texturePerPolygonIdPortionIds.textureHeight;

        // e) portion Id texture for edges
        const texturePerEdgeIdPortionIds = this.generateTextureForPackedPortionIds (
            gl,
            this._portionIdForEdges,
        );

        state.texturePerEdgeIdPortionIds = texturePerEdgeIdPortionIds.texture;
        state.texturePerEdgeIdPortionIdsHeight = texturePerEdgeIdPortionIds.textureHeight;

        // f) indices texture
        const texturePerPolygonIdIndices = this.generateTextureForIndices (
            gl,
            buffer.indices
        );

        state.texturePerPolygonIdIndices = texturePerPolygonIdIndices.texture;
        state.texturePerPolygonIdIndicesHeight = texturePerPolygonIdIndices.textureHeight;
        
        // g) edge indices texture
        const texturePerPolygonIdEdgeIndices = this.generateTextureForEdgeIndices (
            gl,
            buffer.edgeIndices
        );

        state.texturePerPolygonIdEdgeIndices = texturePerPolygonIdEdgeIndices.texture;
        state.texturePerPolygonIdEdgeIndicesHeight = texturePerPolygonIdEdgeIndices.textureHeight;
        
        
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

        state.numIndices = buffer.indices.length;

        state.numEdgeIndices = buffer.edgeIndices.length;

        console.log (JSON.stringify(ramStats, null, 4));

        let totalRamSize = 0;

        Object.keys(ramStats).forEach (key => {
            if (key.startsWith ("size")) {
                totalRamSize+=ramStats[key];
            }
        });

        console.log (`Total size ${totalRamSize} bytes (${(totalRamSize/1000/1000).toFixed(2)} MB)`);

        let percentualRamStats = {};

        Object.keys(ramStats).forEach (key => {
            if (key.startsWith ("size")) {
                percentualRamStats[key] = 
                    `${(ramStats[key] / totalRamSize * 100).toFixed(2)} % of total`;
            }
        });

        console.log (JSON.stringify({percentualRamUsage: percentualRamStats}, null, 4));

        this._buffer = null;
        this._finalized = true;
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

        // 4 columns per texture row:
        // - col0: (RGBA) object color RGBA
        // - col1: (packed Uint32 as RGBA) object pick color
        // - col2: (packed 4 bytes as RGBA) object flags
        // - col3: (packed 4 bytes as RGBA) object flags2
        const textureWidth = 5;

        const texArray = new Uint8Array (4 * textureWidth * textureHeight);

        ramStats.sizeDataColorsAndFlags +=texArray.byteLength;

        for (var i = 0; i < textureHeight; i++)
        {
            // object color
            texArray.set (
                colors [i],
                i * 20 + 0
            );

            // object pick color
            texArray.set (
                pickColors [i],
                i * 20 + 4
            );

            // object flags
            texArray.set (
                [
                    0, 0, 0, 0
                ],
                i * 20 + 8
            );

            // object flags2
            texArray.set (
                [
                    0, 0, 0, 0
                ],
                i * 20 + 12
            );

            // vertex base
            texArray.set (
                [
                    (vertexBases[i] >> 24) & 255,
                    (vertexBases[i] >> 16) & 255,
                    (vertexBases[i] >> 8) & 255,
                    (vertexBases[i]) & 255,
                ],
                i * 20 + 16
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
        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGBA8UI,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGBA_INTEGER,
        //     gl.UNSIGNED_BYTE,
        //     texArray,
        //     0
        // );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
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

        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGBA16F,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGBA,
        //     gl.HALF_FLOAT,
        //     new Uint16Array (texArray.buffer),
        //     0
        // );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
    }

    /**
     * TODO: document
     */
    generateTextureForIndices (gl, indices) {
        const textureWidth = 512;
        const textureHeight = Math.ceil (indices.length / 3 / textureWidth);

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

        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGB16UI,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGB_INTEGER,
        //     gl.UNSIGNED_SHORT,
        //     texArray
        // );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
    }

    /**
     * TODO: comment
     */
    generateTextureForEdgeIndices (gl, edgeIndices) {
        const textureWidth = 512;
        const textureHeight = Math.ceil (edgeIndices.length / 2 / textureWidth);

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

        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGB16UI,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGB_INTEGER,
        //     gl.UNSIGNED_SHORT,
        //     texArray
        // );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
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

        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGB16UI,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGB_INTEGER,
        //     gl.UNSIGNED_SHORT,
        //     texArray    
        // );

        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
    }

    /**
     */
     generateTextureForPackedPortionIds (gl, portionIdsArray) {
        const lenArray = portionIdsArray.length;
        const textureWidth = 512;
        const textureHeight = Math.ceil (
            Math.ceil (lenArray / 2) / // every 2 items will use 3 bytes: 12-bits per item
            textureWidth
        );

        const texArraySize = textureWidth * textureHeight * 3;
        const texArray = new Uint8Array (texArraySize);

        ramStats.sizeDataTexturePortionIds += texArray.byteLength;

        let j = 0;

        for (let i = 0; i < lenArray; i+=2)
        {
            // upper 12 bits contain object Ids for even polygons/edges
            let upper8Bits = portionIdsArray[i] >> 4;
            let half8BitsUpper = portionIdsArray[i] & 15;

            // lower 12 bits contain object Ids for odd polygons/edges
            let half8BitsLower = portionIdsArray[i+1] >> 8;
            let lower8Bits = portionIdsArray[i+1] & 255;

            texArray [j++] = upper8Bits;
            texArray [j++] = (half8BitsUpper << 4) + half8BitsLower;
            texArray [j++] = lower8Bits;
        }

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

        // gl.texImage2D (
        //     gl.TEXTURE_2D,
        //     0,
        //     gl.RGB8UI,
        //     textureWidth,
        //     textureHeight,
        //     0,
        //     gl.RGB_INTEGER,
        //     gl.UNSIGNED_BYTE,
        //     texArray
        // );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);

        gl.bindTexture(gl.TEXTURE_2D, null);

        return {
            texture,
            textureHeight
        };
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

    setCulled(portionId, flags, transparent) {
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
        if (this._state.colorsBuf) {
            this._state.colorsBuf.setData(tempArray, firstColor, lenColor);
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

    _setFlags(portionId, flags, transparent, deferred = false) {
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

        const state = this._state;
        const gl = this.model.scene.canvas.gl;

        gl.bindTexture (gl.TEXTURE_2D, state.texturePerObjectIdColorsAndFlags);

        tempUint8Array4 [0] = f0;
        tempUint8Array4 [1] = f1;
        tempUint8Array4 [2] = f2;
        tempUint8Array4 [3] = f3;

        void gl.texSubImage2D(
            gl.TEXTURE_2D,
            0, // level
            2, // xoffset
            portionId,
            1, // width
            1, //height
            gl.RGBA_INTEGER,
            gl.UNSIGNED_BYTE,
            tempUint8Array4
        );

        gl.bindTexture (gl.TEXTURE_2D, null);
    }

    _setDeferredFlags() {
        if (this._deferredFlagValues) {
            this._state.flagsBuf.setData(this._deferredFlagValues);
            this._deferredFlagValues = null;
        }
    }

    _setFlags2(portionId, flags, deferred = false) {
        if (!this._finalized) {
            throw "Not finalized";
        }

        const clippable = !!(flags & ENTITY_FLAGS.CLIPPABLE) ? 255 : 0;

        const state = this._state;
        const gl = this.model.scene.canvas.gl;

        gl.bindTexture (gl.TEXTURE_2D, state.texturePerObjectIdColorsAndFlags);

        tempUint8Array4 [0] = clippable;
        tempUint8Array4 [1] = 0;
        tempUint8Array4 [2] = 1;
        tempUint8Array4 [3] = 2;

        void gl.texSubImage2D(
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
        gl.bindTexture (gl.TEXTURE_2D, null);
    }

    _setDeferredFlags2() {
        if (this._setDeferredFlag2Values) {
            this._state.flags2Buf.setData(this._setDeferredFlag2Values);
            this._setDeferredFlag2Values = null;
        }
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
            if (frameCtx.pbrEnabled && this.model.pbrEnabled && this._state.normalsBuf) {
                if (this._batchingRenderers.colorQualityRendererWithSAO) {
                    this._batchingRenderers.colorQualityRendererWithSAO.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            } else {
                if (this._state.normalsBuf) {
                    if (this._batchingRenderers.colorRendererWithSAO) {
                        this._batchingRenderers.colorRendererWithSAO.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                    }
                } else {
                    if (this._batchingRenderers.flatColorRendererWithSAO) {
                        this._batchingRenderers.flatColorRendererWithSAO.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                    }
                }
            }
        } else {
            if (frameCtx.pbrEnabled && this.model.pbrEnabled && this._state.normalsBuf) {
                if (this._batchingRenderers.colorQualityRenderer) {
                    this._batchingRenderers.colorQualityRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                }
            } else {
                // if (this._state.normalsBuf) {
                    if (this._batchingRenderers.colorRenderer) {
                        this._batchingRenderers.colorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                    }
                // } else {
                //     if (this._batchingRenderers.flatColorRenderer) {
                //         this._batchingRenderers.flatColorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_OPAQUE);
                //     }
                // }
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
        if (frameCtx.pbrEnabled && this.model.pbrEnabled && this._state.normalsBuf) {
            if (this._batchingRenderers.colorQualityRenderer) {
                this._batchingRenderers.colorQualityRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_TRANSPARENT);
            }
        } else {
            if (this._state.normalsBuf) {
                if (this._batchingRenderers.colorRenderer) {
                    this._batchingRenderers.colorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_TRANSPARENT);
                }
            } else {
                if (this._batchingRenderers.flatColorRenderer) {
                    this._batchingRenderers.flatColorRenderer.drawLayer(frameCtx, this, RENDER_PASSES.COLOR_TRANSPARENT);
                }
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
        if (this._state.normalsBuf) {
            if (this._batchingRenderers.pickNormalsRenderer) {
                this._batchingRenderers.pickNormalsRenderer.drawLayer(frameCtx, this, RENDER_PASSES.PICK);
            }
        } else {
            if (this._batchingRenderers.pickNormalsFlatRenderer) {
                this._batchingRenderers.pickNormalsFlatRenderer.drawLayer(frameCtx, this, RENDER_PASSES.PICK);
            }
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
        if (state.positionsBuf) {
            state.positionsBuf.destroy();
            state.positionsBuf = null;
        }
        if (state.offsetsBuf) {
            state.offsetsBuf.destroy();
            state.offsetsBuf = null;
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
        if (state.pickColorsBuf) {
            state.pickColorsBuf.destroy();
            state.pickColorsBuf = null;
        }
        if (state.indicesBuf) {
            state.indicesBuf.destroy();
            state.indicessBuf = null;
        }
        if (state.edgeIndicesBuf) {
            state.edgeIndicesBuf.destroy();
            state.edgeIndicessBuf = null;
        }
        state.destroy();
    }
}

export {TrianglesBatchingLayer};