import { createRTCViewMat } from "../../../math/rtcCoords.js";

function getNewDataTextureState ()
{
    return {
        /**
         * Texture that holds colors/pickColors/flags/flags2 per-object:
         * - columns: one concept per column => color / pick-color / ...
         * - row: the object Id
         */
        texturePerObjectIdColorsAndFlags: null,
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
    }
}

function generateBindableTexture (gl, texture, textureWidth, textureHeight, textureData = null)
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

function generateCameraDataTexture (gl, camera, scene, origin)
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

    const cameraTexture = generateBindableTexture(
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

function generatePeformanceModelDataTexture (gl, model)
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

    return generateBindableTexture(
        gl,
        texture,
        textureWidth,
        textureHeight
    );
}

export {
    getNewDataTextureState,
    generateBindableTexture,
    generateCameraDataTexture,
    generatePeformanceModelDataTexture,
}