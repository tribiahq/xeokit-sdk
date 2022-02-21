import {WEBGL_INFO} from "../../../../webglInfo.js";

const bigIndicesSupported = WEBGL_INFO.SUPPORTED_EXTENSIONS["OES_element_index_uint"];

/**
 * @private
 */
class TrianglesBatchingBuffer {

    constructor(maxGeometryBatchSize = 5000000) {

        if (bigIndicesSupported) {
            if (maxGeometryBatchSize > 5000000) {
                maxGeometryBatchSize = 5000000;
            }
        } else {
            if (maxGeometryBatchSize > 65530) {
                maxGeometryBatchSize = 65530;
            }
        }

        this.maxVerts = maxGeometryBatchSize;
        this.maxIndices = maxGeometryBatchSize * 3; // Rough rule-of-thumb
        this.positions = [];
        // this.colors = [];
        this.metallicRoughness = [];
        this.normals = [];
        // this.pickColors = [];
        // this.flags = [];
        // this.flags2 = [];
        this.offsets = [];
        this.indices8Bits = [];
        this.indices16Bits = [];
        this.indices32Bits = [];
        this.edgeIndices8Bits = [];
        this.edgeIndices16Bits = [];
        this.edgeIndices32Bits = [];
        this.objectData = [];
    }
}

export {TrianglesBatchingBuffer};