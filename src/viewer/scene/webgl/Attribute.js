/**
 * @desc Represents a WebGL vertex attribute buffer (VBO).
 * @private
 * @param gl {WebGLRenderingContext} The WebGL rendering context.
 */
class Attribute {

    constructor(gl, location) {
        this._gl = gl;
        this.location = location;
    }

    bindArrayBuffer(arrayBuf) {
        if (!arrayBuf) {
            return;
        }
        arrayBuf.bind();
        // chipmunk: not use "I" for non-integer buffers
        this._gl.vertexAttribIPointer(this.location, arrayBuf.itemSize, arrayBuf.itemType, arrayBuf.normalized, arrayBuf.stride, arrayBuf.offset);
        this._gl.enableVertexAttribArray(this.location);
    }
}

export {Attribute};
