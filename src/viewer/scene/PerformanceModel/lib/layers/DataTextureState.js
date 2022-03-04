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

export { getNewDataTextureState }