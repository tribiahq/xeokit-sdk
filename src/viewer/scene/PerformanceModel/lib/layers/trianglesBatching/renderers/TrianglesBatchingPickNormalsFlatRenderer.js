import {Program} from "../../../../../webgl/Program.js";
import {createRTCViewMat, getPlaneRTCPos} from "../../../../../math/rtcCoords.js";
import {math} from "../../../../../math/math.js";
import {WEBGL_INFO} from "../../../../../webglInfo.js";

const tempVec3a = math.vec3();

/**
 * @private
 */
class TrianglesBatchingPickNormalsFlatRenderer {

    constructor(scene) {
        this._scene = scene;
        this._hash = this._getHash();
        this._allocate();
    }

    getValid() {
        return this._hash === this._getHash();
    };

    _getHash() {
        return this._scene._sectionPlanesState.getHash();
    }

    drawLayer(frameCtx, batchingLayer, renderPass) {

        const model = batchingLayer.model;
        const scene = model.scene;
        const camera = scene.camera;
        const gl = scene.canvas.gl;
        const state = batchingLayer._state;
        const origin = batchingLayer._state.origin;

        if (!this._program) {
            this._allocate(batchingLayer);
        }

        if (frameCtx.lastProgramId !== this._program.id) {
            frameCtx.lastProgramId = this._program.id;
            this._bindProgram();
        }

        var rr = this._program.bindTexture(
            this._uTexturePerObjectPositionsDecodeMatrix, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerObjectPositionsDecodeMatrix);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            1
        ); // chipmunk

        var rr2 = this._program.bindTexture(
            this._uPositionsTexture, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.positionsTexture);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            2
        ); // chipmunk

        var rr3 = this._program.bindTexture(
            this._uNormalsPerPolygonTexture, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.normalsPerPolygonTexture);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            3
        ); // chipmunk

        var rr4 = this._program.bindTexture(
            this._uTexturePerObjectColorsAndFlags,
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerObjectColorsAndFlags);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            4
        ); // chipmunk

        gl.uniform1i(this._uTexturePerObjectColorsAndFlagsHeight, state.texturePerObjectColorsAndFlagsHeight);

        gl.uniform1i(this._uRenderPass, renderPass);
        gl.uniform1i(this._uPickInvisible, frameCtx.pickInvisible);

        gl.uniformMatrix4fv(this._uWorldMatrix, false, model.worldMatrix);

        const pickViewMatrix = frameCtx.pickViewMatrix || camera.viewMatrix;
        const viewMatrix = origin ? createRTCViewMat(pickViewMatrix, origin) : pickViewMatrix;

        gl.uniformMatrix4fv(this._uViewMatrix, false, viewMatrix);
        gl.uniformMatrix4fv(this._uProjMatrix, false, frameCtx.pickProjMatrix);

        if (scene.logarithmicDepthBufferEnabled) {
            const logDepthBufFC = 2.0 / (Math.log(camera.project.far + 1.0) / Math.LN2);  // TODO: Far should be from projection matrix?
            gl.uniform1f(this._uLogDepthBufFC, logDepthBufFC);
        }

        const numSectionPlanes = scene._sectionPlanesState.sectionPlanes.length;
        if (numSectionPlanes > 0) {
            const sectionPlanes = scene._sectionPlanesState.sectionPlanes;
            const baseIndex = batchingLayer.layerIndex * numSectionPlanes;
            const renderFlags = model.renderFlags;
            for (let sectionPlaneIndex = 0; sectionPlaneIndex < numSectionPlanes; sectionPlaneIndex++) {
                const sectionPlaneUniforms = this._uSectionPlanes[sectionPlaneIndex];
                const active = renderFlags.sectionPlanesActivePerLayer[baseIndex + sectionPlaneIndex];
                gl.uniform1i(sectionPlaneUniforms.active, active ? 1 : 0);
                if (active) {
                    const sectionPlane = sectionPlanes[sectionPlaneIndex];
                    if (origin) {
                        const rtcSectionPlanePos = getPlaneRTCPos(sectionPlane.dist, sectionPlane.dir, origin, tempVec3a);
                        gl.uniform3fv(sectionPlaneUniforms.pos, rtcSectionPlanePos);
                    } else {
                        gl.uniform3fv(sectionPlaneUniforms.pos, sectionPlane.pos);
                    }
                    gl.uniform3fv(sectionPlaneUniforms.dir, sectionPlane.dir);
                }
            }
        }

        //=============================================================
        // TODO: Use drawElements count and offset to draw only one entity
        //=============================================================

        if (this._aPackedVertexId) {
            this._aPackedVertexId.bindArrayBuffer(state.indicesBuf);
        }


        gl.drawArrays(gl.TRIANGLES, 0, state.indicesBuf.numItems);

        frameCtx.drawElements++;
    }

    _allocate() {

        const scene = this._scene;
        const gl = scene.canvas.gl;

        this._program = new Program(gl, this._buildShader());

        if (this._program.errors) {
            this.errors = this._program.errors;
            return;
        }

        const program = this._program;

        this._uRenderPass = program.getLocation("renderPass");
        this._uPickInvisible = program.getLocation("pickInvisible");
        this._uPositionsDecodeMatrix = program.getLocation("positionsDecodeMatrix");
        this._uWorldMatrix = program.getLocation("worldMatrix");
        this._uViewMatrix = program.getLocation("viewMatrix");
        this._uProjMatrix = program.getLocation("projMatrix");
        this._uSectionPlanes = [];

        for (let i = 0, len = scene._sectionPlanesState.sectionPlanes.length; i < len; i++) {
            this._uSectionPlanes.push({
                active: program.getLocation("sectionPlaneActive" + i),
                pos: program.getLocation("sectionPlanePos" + i),
                dir: program.getLocation("sectionPlaneDir" + i)
            });
        }

        this._aPackedVertexId = program.getAttribute("packedVertexId");


        if (scene.logarithmicDepthBufferEnabled) {
            this._uLogDepthBufFC = program.getLocation("logDepthBufFC");
        }

        this._uTexturePerObjectPositionsDecodeMatrix = "uTexturePerObjectPositionsDecodeMatrix"; // chipmunk
        this._uTexturePerObjectColorsAndFlags = "uTexturePerObjectColorsAndFlags"; // chipmunk
        this._uPositionsTexture = "uPositionsTexture"; // chipmunk
        this._uNormalsPerPolygonTexture = "uNormalsPerPolygonTexture"; // chipmunk
    }

    _bindProgram() {
        this._program.bind();
    }

    _buildShader() {
        return {
            vertex: this._buildVertexShader(),
            fragment: this._buildFragmentShader()
        };
    }

    _buildVertexShader() {
        const scene = this._scene;
        const clipping = scene._sectionPlanesState.sectionPlanes.length > 0;
        const src = [];
        src.push("#version 300 es");
        src.push("// Triangles batching pick flat normals vertex shader");
        if (scene.logarithmicDepthBufferEnabled && WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
            src.push("#extension GL_EXT_frag_depth : enable");
        }

        src.push("#ifdef GL_FRAGMENT_PRECISION_HIGH");
        src.push("precision highp float;");
        src.push("precision highp int;");
        src.push("precision highp usampler2D;");
        src.push("precision highp isampler2D;");
        src.push("precision highp sampler2D;");
        src.push("#else");
        src.push("precision mediump float;");
        src.push("precision mediump int;");
        src.push("precision mediump usampler2D;");
        src.push("precision mediump isampler2D;");
        src.push("precision mediump sampler2D;");
        src.push("#endif");

        src.push("uniform int renderPass;");
        src.push("uniform highp int texturePerObjectColorsAndFlagsHeight;");

        src.push("in uvec3 packedVertexId;");


        if (scene.entityOffsetsEnabled) {
            src.push("in vec3 offset;");
        }

        src.push("uniform bool pickInvisible;");
        src.push("uniform mat4 worldMatrix;");
        src.push("uniform mat4 viewMatrix;");
        src.push("uniform mat4 projMatrix;");
        // src.push("uniform sampler2D uOcclusionTexture;"); // chipmunk
        src.push("uniform sampler2D uTexturePerObjectPositionsDecodeMatrix;"); // chipmunk
        src.push("uniform usampler2D uTexturePerObjectColorsAndFlags;"); // chipmunk
        src.push("uniform usampler2D uPositionsTexture;"); // chipmunk
        src.push("uniform isampler2D uNormalsPerPolygonTexture;"); // chipmunk

        if (scene.logarithmicDepthBufferEnabled) {
            src.push("uniform float logDepthBufFC;");
            if (WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
                src.push("varying float vFragDepth;");
            }
            src.push("bool isPerspectiveMatrix(mat4 m) {");
            src.push("    return (m[2][3] == - 1.0);");
            src.push("}");
            src.push("varying float isPerspective;");
        }
        src.push("out vec4 vWorldPosition;");
        if (clipping) {
            src.push("out int vFlags2;");
        }
        src.push("void main(void) {");

        // constants
        src.push("int objectIndex = int(packedVertexId.g) & 1023;");
        src.push("int polygonIndex = gl_VertexID / 3;")
        src.push("int uniqueVertexIndex = int ((packedVertexId.r << 6) + (packedVertexId.g >> 10));");


        src.push("int h_unique_position_index = uniqueVertexIndex & 511;")
        src.push("int v_unique_position_index = uniqueVertexIndex >> 9;")

        src.push("mat4 positionsDecodeMatrix = mat4 (texelFetch (uTexturePerObjectPositionsDecodeMatrix, ivec2(0, objectIndex), 0), texelFetch (uTexturePerObjectPositionsDecodeMatrix, ivec2(1, objectIndex), 0), texelFetch (uTexturePerObjectPositionsDecodeMatrix, ivec2(2, objectIndex), 0), texelFetch (uTexturePerObjectPositionsDecodeMatrix, ivec2(3, objectIndex), 0));")

        // get flags & flags2
        src.push("uvec4 flags = texelFetch (uTexturePerObjectColorsAndFlags, ivec2(2, objectIndex), 0);"); // chipmunk
        src.push("uvec4 flags2 = texelFetch (uTexturePerObjectColorsAndFlags, ivec2(3, objectIndex), 0);"); // chipmunk
        
        // get position
        src.push("vec3 position = vec3(texelFetch(uPositionsTexture, ivec2(h_unique_position_index, v_unique_position_index), 0).rgb);")

        // flags.w = NOT_RENDERED | PICK
        // renderPass = PICK
        src.push(`if (int(flags.w) != renderPass) {`);
        src.push("      gl_Position = vec4(0.0, 0.0, 0.0, 0.0);"); // Cull vertex
        src.push("  } else {");
        src.push("      vec4 worldPosition = worldMatrix * (positionsDecodeMatrix * vec4(position, 1.0)); ");
        if (scene.entityOffsetsEnabled) {
            src.push("      worldPosition.xyz = worldPosition.xyz + offset;");
        }
        src.push("      vec4 viewPosition  = viewMatrix * worldPosition; ");
        src.push("      vWorldPosition = worldPosition;");
        if (clipping) {
            src.push("      vFlags2 = flags2.r;");
        }
        src.push("vec4 clipPos = projMatrix * viewPosition;");
        if (scene.logarithmicDepthBufferEnabled) {
            if (WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
                src.push("vFragDepth = 1.0 + clipPos.w;");
            } else {
                src.push("clipPos.z = log2( max( 1e-6, clipPos.w + 1.0 ) ) * logDepthBufFC - 1.0;");
                src.push("clipPos.z *= clipPos.w;");
            }
            src.push("isPerspective = float (isPerspectiveMatrix(projMatrix));");
        }
        src.push("gl_Position = clipPos;");
        src.push("  }");
        src.push("}");
        return src;
    }

    _buildFragmentShader() {
        const scene = this._scene;
        const sectionPlanesState = scene._sectionPlanesState;
        const clipping = sectionPlanesState.sectionPlanes.length > 0;
        const src = [];
        src.push ('#version 300 es');
        src.push("// Triangles batching pick flat normals fragment shader");
        src.push("#extension GL_OES_standard_derivatives : enable");
        if (scene.logarithmicDepthBufferEnabled && WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
            src.push("#extension GL_EXT_frag_depth : enable");
        }
        src.push("#ifdef GL_FRAGMENT_PRECISION_HIGH");
        src.push("precision highp float;");
        src.push("precision highp int;");
        src.push("#else");
        src.push("precision mediump float;");
        src.push("precision mediump int;");
        src.push("#endif");
        if (scene.logarithmicDepthBufferEnabled && WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
            src.push("varying float isPerspective;");
            src.push("uniform float logDepthBufFC;");
            src.push("in float vFragDepth;");
        }
        src.push("in vec4 vWorldPosition;");
        if (clipping) {
            src.push("in int vFlags2;");
            for (var i = 0; i < sectionPlanesState.sectionPlanes.length; i++) {
                src.push("uniform bool sectionPlaneActive" + i + ";");
                src.push("uniform vec3 sectionPlanePos" + i + ";");
                src.push("uniform vec3 sectionPlaneDir" + i + ";");
            }
        }
        src.push("out vec4 outNormal;");
        src.push("void main(void) {");
        if (clipping) {
            src.push("  bool clippable = vFlags2 > 0;");
            src.push("  if (clippable) {");
            src.push("      float dist = 0.0;");
            for (var i = 0; i < sectionPlanesState.sectionPlanes.length; i++) {
                src.push("      if (sectionPlaneActive" + i + ") {");
                src.push("          dist += clamp(dot(-sectionPlaneDir" + i + ".xyz, vWorldPosition.xyz - sectionPlanePos" + i + ".xyz), 0.0, 1000.0);");
                src.push("      }");
            }
            src.push("      if (dist > 0.0) { discard; }");
            src.push("  }");
        }
        if (scene.logarithmicDepthBufferEnabled && WEBGL_INFO.SUPPORTED_EXTENSIONS["EXT_frag_depth"]) {
            src.push("    gl_FragDepthEXT = isPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;");
        }
        src.push("  vec3 xTangent = dFdx( vWorldPosition.xyz );");
        src.push("  vec3 yTangent = dFdy( vWorldPosition.xyz );");
        src.push("  vec3 worldNormal = normalize( cross( xTangent, yTangent ) );");
        src.push("  outNormal = vec4((worldNormal * 0.5) + 0.5, 1.0);");
        src.push("}");
        return src;
    }

    webglContextRestored() {
        this._program = null;
    }

    destroy() {
        if (this._program) {
            this._program.destroy();
        }
        this._program = null;
    }
}

export {TrianglesBatchingPickNormalsFlatRenderer};