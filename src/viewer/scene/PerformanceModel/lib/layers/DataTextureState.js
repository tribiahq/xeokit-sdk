import { createRTCViewMat } from "../../../math/rtcCoords.js";
import { Float16Array, isFloat16Array, getFloat16, setFloat16, hfround, } from "./float16.js";

const ramStats = {
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

class BindableDataTexture
{
    /**
     * 
     * @param {WebGL2RenderingContext} gl 
     * @param {WebGLTexture} texture 
     * @param {int} textureWidth 
     * @param {int} textureHeight 
     * @param {TypedArray} textureData 
     */
    constructor(gl, texture, textureWidth, textureHeight, textureData = null)
    {
        /**
         * The WebGL context.
         * 
         * @type WebGL2RenderingContext
         * @private
         */
        this._gl = gl;

        /**
         * The WebGLTexture handle.
         * 
         * @type WebGLTexture
         * @private
         */
        this._texture = texture;

        /**
         * The texture width.
         * 
         * @type int
         * @private
         */
        this._textureWidth = textureWidth;

        /**
         * The texture height.
         * 
         * @type int
         * @private
         */
        this._textureHeight = textureHeight;

         /**
          * (nullable) When the texture data array is kept in the JS side, it will be stored here.
          * 
          * @type TypedArray
          * @private
          */
        this._textureData = textureData;
    }

    /**
     * Convenience method to be used by the renderers to bind the texture before draw calls.
     * 
     * @returns {bool}
     * 
     * @public
     */
    bindTexture (glProgram, shaderName, glTextureUnit) {
        return glProgram.bindTexture (shaderName, this, glTextureUnit);
    }

    /**
     * Used internally by the `program` passed to `bindTexture` in order to bind the texture to an active `texture-unit`.
     * 
     * @returns {bool}
     * @private
     */
    bind (unit) {
        this._gl.activeTexture(this._gl["TEXTURE" + unit]);
        this._gl.bindTexture(this._gl.TEXTURE_2D, this._texture);
        return true;
    }

    /**
     * Used internally by the `program` passed to `bindTexture` in order to bind the texture to an active `texture-unit`.
     * 
     * @private
     */
    unbind (unit) {
        // This `unbind` method is ignored at the moment to allow avoiding to rebind same texture already bound to a texture unit.

        // this._gl.activeTexture(this.state.gl["TEXTURE" + unit]);
        // this._gl.bindTexture(this.state.gl.TEXTURE_2D, null);
    }
}

class DataTextureState
{
    constructor ()
    {
        /**
         * Texture that holds colors/pickColors/flags/flags2 per-object:
         * - columns: one concept per column => color / pick-color / ...
         * - row: the object Id
         * 
         * @type BindableDataTexture
         */
        this.texturePerObjectIdColorsAndFlags = null;

        /**
         * Texture that holds the positionsDecodeMatrix per-object:
         * - columns: each column is one column of the matrix
         * - row: the object Id
         * 
         * @type BindableDataTexture
         */
        this.texturePerObjectIdPositionsDecodeMatrix = null;

        /**
         * Texture that holds all the `different-vertices` used by the layer.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerVertexIdCoordinates = null;

        /**
         * Texture that holds the PortionId that corresponds to a given polygon-id.
         * 
         * Variant of the texture for 8-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerPolygonIdPortionIds8Bits = null;

        /**
         * Texture that holds the PortionId that corresponds to a given polygon-id.
         * 
         * Variant of the texture for 16-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerPolygonIdPortionIds16Bits = null;

        /**
         * Texture that holds the PortionId that corresponds to a given polygon-id.
         * 
         * Variant of the texture for 32-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerPolygonIdPortionIds32Bits = null;

        /**
         * Texture that holds the PortionId that corresponds to a given edge-id.
         * 
         * Variant of the texture for 8-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerEdgeIdPortionIds8Bits = null;

        /**
         * Texture that holds the PortionId that corresponds to a given edge-id.
         * 
         * Variant of the texture for 16-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerEdgeIdPortionIds16Bits = null;

        /**
         * Texture that holds the PortionId that corresponds to a given edge-id.
         * 
         * Variant of the texture for 32-bit based polygon-ids.
         * 
         * @type BindableDataTexture
         */
        this.texturePerEdgeIdPortionIds32Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 8-bit based indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdIndices8Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 16-bit based indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdIndices16Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 32-bit based indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdIndices32Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 8-bit based edge indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdEdgeIndices8Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 16-bit based edge indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdEdgeIndices16Bits = null;

        /**
         * Texture that holds the unique-vertex-indices for 32-bit based edge indices.
         * 
         * @type BindableDataTexture
         */            
        this.texturePerPolygonIdEdgeIndices32Bits = null;

        /**
         * Texture that holds the camera matrices
         * - columns: each column in the texture is a camera matrix column.
         * - row: each row is a different camera matrix.
         * 
         * @type BindableDataTexture
         */
        this.textureCameraMatrices = null;

        /**
         * Texture that holds the model matrices
         * - columns: each column in the texture is a model matrix column.
         * - row: each row is a different model matrix.
         * 
         * @type BindableDataTexture
         */
        this.textureModelMatrices = null;
    }
}

class DataTextureGenerator
{
    /**
     * Generate and return a `camera data texture`.
     * 
     * The texture will automatically update its contents before each render when the camera matrix is dirty. 
     * 
     * @param {WebGL2RenderingContext} gl 
     * @param {*} camera 
     * @param {*} scene 
     * @param {*} origin 
     * 
     * @returns {BindableDataTexture}
     */
    generateCameraDataTexture (gl, camera, scene, origin)
    {
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

        const cameraTexture = new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );

        let cameraDirty = true;

        const onCameraMatrix = () => {
            if (!cameraDirty) {
                return;
            }

            cameraDirty = false;
            
            gl.bindTexture (gl.TEXTURE_2D, cameraTexture._texture);

            // Camera's "view matrix"
            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                0, // 1st matrix: camera view matrix
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
                1, // 2nd matrix: camera view normal matrix
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
                2, // 3rd matrix: camera project matrix
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

        return cameraTexture;
    }

    /**
     * Generate and return a `model data texture`.
     *
     * @param {WebGL2RenderingContext} gl 
     * @param {PerformanceModel} model 
     * 
     * @returns {BindableDataTexture}
    */
    generatePeformanceModelDataTexture (gl, model)
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
            new Float32Array (model.worldMatrix)
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
            new Float32Array (model.worldNormalMatrix)
        );

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.bindTexture (gl.TEXTURE_2D, null);

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * This will generate an RGBA texture for:
     * - colors
     * - pickColors
     * - flags
     * - flags2
     * - vertex bases
     * 
     * The texture will have:
     * - 4 RGBA columns per row: for each object (pick) color and flags(2)
     * - N rows where N is the number of objects
     * 
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<ArrayLike<int>>} colors Array of colors for all objects in the layer
     * @param {ArrayLike<ArrayLike<int>>} pickColors Array of pickColors for all objects in the layer
     * @param {ArrayLike<int>} vertexBases Array of position-index-bases foteh all objects in the layer
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
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
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<Matrix4x4>} positionDecodeMatrices Array of positions decode matrices for all objects in the layer
     * @param {ArrayLike<Matrix4x4>} instanceMatrices Array of geometry instancing matrices for all objects in the layer. Null if the objects are not instanced.
     * @param {ArrayLike<Matrix4x4>} instancesNormalMatrices Array of normals instancing matrices for all objects in the layer. Null if the objects are not instanced.
     * 
     * @returns {BindableDataTexture}
     */
    generateTextureForPositionsDecodeMatrices (gl, positionDecodeMatrices, instanceMatrices = null, instancesNormalMatrices = null) {
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} indices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} indices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} indices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} edgeIndices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} edgeIndices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} edgeIndices
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} positions Array of (uniquified) quantized positions in the layer
     * 
     * This will generate a texture for positions in the layer.
     * 
     * The texture will have:
     * - 512 columns, where each pixel will be a 16-bit-per-component RGB texture, corresponding to the XYZ of the position 
     * - a number of rows R where R*512 is just >= than the number of vertices (positions / 3)
     * 
     * @returns {BindableDataTexture}
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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @param {ArrayLike<int>} portionIdsArray
     * 
     * @returns {BindableDataTexture}
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
        const textureHeight = Math.ceil (lenArray / textureWidth);

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

        return new BindableDataTexture(
            gl,
            texture,
            textureWidth,
            textureHeight
        );
    }
}

export {
    ramStats,
    DataTextureState,
    DataTextureGenerator,
}