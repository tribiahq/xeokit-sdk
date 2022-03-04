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

export { getNewDataTextureState, generateBindableTexture }