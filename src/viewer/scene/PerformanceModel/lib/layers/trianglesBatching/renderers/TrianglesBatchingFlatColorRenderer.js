import {Program} from "../../../../../webgl/Program.js";
import {math} from "../../../../../math/math.js";
import {createRTCViewMat, getPlaneRTCPos} from "../../../../../math/rtcCoords.js";
import {WEBGL_INFO} from "../../../../../webglInfo.js";

const tempVec4 = math.vec4();
const tempVec3a = math.vec3();

/**
 * @private
 */
class TrianglesBatchingFlatColorRenderer {

    constructor(scene, withSAO) {
        this._scene = scene;
        this._withSAO = withSAO;
        this._hash = this._getHash();
        this._allocate();
    }

    getValid() {
        return this._hash === this._getHash();
    };

    _getHash() {
        const scene = this._scene;
        return [scene._lightsState.getHash(), scene._sectionPlanesState.getHash(), (this._withSAO ? "sao" : "nosao")].join(";");
    }

    drawLayer(frameCtx, batchingLayer, renderPass) {
        const scene = this._scene;
        const camera = scene.camera;
        const model = batchingLayer.model;
        const gl = scene.canvas.gl;
        const state = batchingLayer._state;
        const origin = batchingLayer._state.origin;

        if (!this._program) {
            this._allocate();
            if (this.errors) {
                return;
            }
        }

        if (frameCtx.lastProgramId !== this._program.id) {
            frameCtx.lastProgramId = this._program.id;
            this._bindProgram(frameCtx);
        }

        var rr = this._program.bindTexture(
            this._uTexturePerObjectIdPositionsDecodeMatrix, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerObjectIdPositionsDecodeMatrix);
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
            this._uTexturePerVertexIdCoordinates, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerVertexIdCoordinates);
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
            this._uTexturePerPolygonIdNormals, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerPolygonIdNormals);
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
            this._uTexturePerObjectIdColorsAndFlags,
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerObjectIdColorsAndFlags);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            4
        ); // chipmunk

        var rr5 = this._program.bindTexture(
            this._uTexturePerPolygonIdPortionIds, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerPolygonIdPortionIds);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            5
        ); // chipmunk

        var rr6 = this._program.bindTexture(
            this._uTexturePerPolygonIdIndices, 
            {
                bind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, state.texturePerPolygonIdIndices);
                    return true;
                },
                unbind: function (unit) {
                    gl.activeTexture(gl["TEXTURE" + unit]);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
            },
            6
        ); // chipmunk

        gl.uniform1i(this._uTexturePerObjectIdColorsAndFlagsHeight, state.texturePerObjectIdColorsAndFlagsHeight);

        gl.uniform1i(this._uRenderPass, renderPass);

        gl.uniformMatrix4fv(this._uViewMatrix, false, (origin) ? createRTCViewMat(camera.viewMatrix, origin) : camera.viewMatrix);

        gl.uniformMatrix4fv(this._uWorldMatrix, false, model.worldMatrix);

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


        gl.drawArrays(gl.TRIANGLES, 0, state.numIndices);
    }

    _allocate() {

        const scene = this._scene;
        const gl = scene.canvas.gl;
        const lightsState = scene._lightsState;

        this._program = new Program(gl, this._buildShader());

        if (this._program.errors) {
            this.errors = this._program.errors;
            return;
        }

        const program = this._program;

        this._uTexturePerObjectIdColorsAndFlagsHeight = program.getLocation("texturePerObjectIdColorsAndFlagsHeight");
        this._uRenderPass = program.getLocation("renderPass");
        this._uWorldMatrix = program.getLocation("worldMatrix");
        this._uWorldNormalMatrix = program.getLocation("worldNormalMatrix");
        this._uViewMatrix = program.getLocation("viewMatrix");
        this._uViewNormalMatrix = program.getLocation("viewNormalMatrix");
        this._uProjMatrix = program.getLocation("projMatrix");

        this._uLightAmbient = program.getLocation("lightAmbient");
        this._uLightColor = [];
        this._uLightDir = [];
        this._uLightPos = [];
        this._uLightAttenuation = [];

        const lights = lightsState.lights;
        let light;

        for (let i = 0, len = lights.length; i < len; i++) {
            light = lights[i];
            switch (light.type) {
                case "dir":
                    this._uLightColor[i] = program.getLocation("lightColor" + i);
                    this._uLightPos[i] = null;
                    this._uLightDir[i] = program.getLocation("lightDir" + i);
                    break;
                case "point":
                    this._uLightColor[i] = program.getLocation("lightColor" + i);
                    this._uLightPos[i] = program.getLocation("lightPos" + i);
                    this._uLightDir[i] = null;
                    this._uLightAttenuation[i] = program.getLocation("lightAttenuation" + i);
                    break;
                case "spot":
                    this._uLightColor[i] = program.getLocation("lightColor" + i);
                    this._uLightPos[i] = program.getLocation("lightPos" + i);
                    this._uLightDir[i] = program.getLocation("lightDir" + i);
                    this._uLightAttenuation[i] = program.getLocation("lightAttenuation" + i);
                    break;
            }
        }

        this._uSectionPlanes = [];

        for (let i = 0, len = scene._sectionPlanesState.sectionPlanes.length; i < len; i++) {
            this._uSectionPlanes.push({
                active: program.getLocation("sectionPlaneActive" + i),
                pos: program.getLocation("sectionPlanePos" + i),
                dir: program.getLocation("sectionPlaneDir" + i)
            });
        }
        
        this._aPackedVertexId = program.getAttribute("packedVertexId");

        if (this._withSAO) {
            this._uOcclusionTexture = "uOcclusionTexture";
            this._uSAOParams = program.getLocation("uSAOParams");
        }

        if (scene.logarithmicDepthBufferEnabled) {
            this._uLogDepthBufFC = program.getLocation("logDepthBufFC");
        }

        this._uTexturePerObjectIdPositionsDecodeMatrix = "uTexturePerObjectIdPositionsDecodeMatrix"; // chipmunk
        this._uTexturePerObjectIdColorsAndFlags = "uTexturePerObjectIdColorsAndFlags"; // chipmunk
        this._uTexturePerVertexIdCoordinates = "uTexturePerVertexIdCoordinates"; // chipmunk
        this._uTexturePerPolygonIdNormals = "uTexturePerPolygonIdNormals"; // chipmunk
        this._uTexturePerPolygonIdIndices = "uTexturePerPolygonIdIndices"; // chipmunk
        this._uTexturePerPolygonIdPortionIds = "uTexturePerPolygonIdPortionIds"; // chipmunk
    }

    _bindProgram(frameCtx) {

        const scene = this._scene;
        const gl = scene.canvas.gl;
        const program = this._program;
        const lights = scene._lightsState.lights;
        const project = scene.camera.project;

        program.bind();

        gl.uniformMatrix4fv(this._uProjMatrix, false, project.matrix)

        if (this._uLightAmbient) {
            gl.uniform4fv(this._uLightAmbient, scene._lightsState.getAmbientColorAndIntensity());
        }

        for (let i = 0, len = lights.length; i < len; i++) {
            const light = lights[i];

            if (this._uLightColor[i]) {
                gl.uniform4f(this._uLightColor[i], light.color[0], light.color[1], light.color[2], light.intensity);
            }
            if (this._uLightPos[i]) {
                gl.uniform3fv(this._uLightPos[i], light.pos);
                if (this._uLightAttenuation[i]) {
                    gl.uniform1f(this._uLightAttenuation[i], light.attenuation);
                }
            }
            if (this._uLightDir[i]) {
                gl.uniform3fv(this._uLightDir[i], light.dir);
            }
        }

        if (this._withSAO) {
            const sao = scene.sao;
            const saoEnabled = sao.possible;
            if (saoEnabled) {
                const viewportWidth = gl.drawingBufferWidth;
                const viewportHeight = gl.drawingBufferHeight;
                tempVec4[0] = viewportWidth;
                tempVec4[1] = viewportHeight;
                tempVec4[2] = sao.blendCutoff;
                tempVec4[3] = sao.blendFactor;
                gl.uniform4fv(this._uSAOParams, tempVec4);
                this._program.bindTexture(this._uOcclusionTexture, frameCtx.occlusionTexture, 0);
            }
        }

        if (scene.logarithmicDepthBufferEnabled) {
            const logDepthBufFC = 2.0 / (Math.log(project.far + 1.0) / Math.LN2);
            gl.uniform1f(this._uLogDepthBufFC, logDepthBufFC);
        }
    }

    _buildShader() {
        return {
            vertex: this._buildVertexShader(),
            fragment: this._buildFragmentShader()
        };
    }

    _buildVertexShader() {
        const scene = this._scene;
        const sectionPlanesState = scene._sectionPlanesState;
        const clipping = sectionPlanesState.sectionPlanes.length > 0;
        const src = [];
        src.push("#version 300 es");
        src.push("// Triangles batching flat-shading draw vertex shader");

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
        src.push("uniform highp int texturePerObjectIdColorsAndFlagsHeight;");

        src.push("in uvec3 packedVertexId;");


        if (scene.entityOffsetsEnabled) {
            src.push("in vec3 offset;");
        }

        src.push("uniform mat4 worldMatrix;");

        src.push("uniform mat4 viewMatrix;");
        src.push("uniform mat4 projMatrix;");
        // src.push("uniform sampler2D uOcclusionTexture;"); // chipmunk
        src.push("uniform sampler2D uTexturePerObjectIdPositionsDecodeMatrix;"); // chipmunk
        src.push("uniform usampler2D uTexturePerObjectIdColorsAndFlags;"); // chipmunk
        src.push("uniform usampler2D uTexturePerVertexIdCoordinates;"); // chipmunk
        src.push("uniform usampler2D uTexturePerPolygonIdIndices;"); // chipmunk
        src.push("uniform isampler2D uTexturePerPolygonIdNormals;"); // chipmunk
        src.push("uniform usampler2D uTexturePerPolygonIdPortionIds;"); // chipmunk

        if (scene.logarithmicDepthBufferEnabled) {
            src.push("uniform float logDepthBufFC;");
            src.push("out float vFragDepth;");
            src.push("bool isPerspectiveMatrix(mat4 m) {");
            src.push("    return (m[2][3] == - 1.0);");
            src.push("}");
            src.push("out float isPerspective;");
        }
        
        if (clipping) {
            src.push("out vec4 vWorldPosition;");
            src.push("out int vFlags2;");
        }
        
        src.push("out vec4 vViewPosition;");
        src.push("out vec4 vColor;");

        src.push("void main(void) {");

        // constants
        src.push("int polygonIndex = gl_VertexID / 3;")

        src.push("int h_normal_index = polygonIndex & 511;")
        src.push("int v_normal_index = polygonIndex >> 9;")

        // get packed object-id
        src.push("int h_packed_object_id_index = ((polygonIndex >> 3) / 2) & 511;")
        src.push("int v_packed_object_id_index = ((polygonIndex >> 3) / 2) >> 9;")

        src.push("ivec3 packedObjectId = ivec3(texelFetch(uTexturePerPolygonIdPortionIds, ivec2(h_packed_object_id_index, v_packed_object_id_index), 0).rgb);");

        src.push("int objectIndex;")
        src.push("if (((polygonIndex >> 3) % 2) == 0) {")
        src.push("  objectIndex = (packedObjectId.r << 4) + (packedObjectId.g >> 4);")
        src.push("} else {") 
        src.push("  objectIndex = ((packedObjectId.g & 15) << 8) + packedObjectId.b;")
        src.push("}")

        // get vertex base
        src.push("ivec4 packedVertexBase = ivec4(texelFetch (uTexturePerObjectIdColorsAndFlags, ivec2(4, objectIndex), 0));"); // chipmunk

        src.push("int h_index = polygonIndex & 511;")
        src.push("int v_index = polygonIndex >> 9;")

        src.push("ivec3 vertexIndices = ivec3(texelFetch(uTexturePerPolygonIdIndices, ivec2(h_index, v_index), 0));");
        src.push("ivec3 uniqueVertexIndexes = vertexIndices + (packedVertexBase.r << 24) + (packedVertexBase.g << 16) + (packedVertexBase.b << 8) + packedVertexBase.a;")
        
        src.push("ivec3 indexPositionH = uniqueVertexIndexes & 511;")
        src.push("ivec3 indexPositionV = uniqueVertexIndexes >> 9;")

        src.push("mat4 positionsDecodeMatrix = mat4 (texelFetch (uTexturePerObjectIdPositionsDecodeMatrix, ivec2(0, objectIndex), 0), texelFetch (uTexturePerObjectIdPositionsDecodeMatrix, ivec2(1, objectIndex), 0), texelFetch (uTexturePerObjectIdPositionsDecodeMatrix, ivec2(2, objectIndex), 0), texelFetch (uTexturePerObjectIdPositionsDecodeMatrix, ivec2(3, objectIndex), 0));")

        // get flags & flags2
        src.push("uvec4 flags = texelFetch (uTexturePerObjectIdColorsAndFlags, ivec2(2, objectIndex), 0);"); // chipmunk
        src.push("uvec4 flags2 = texelFetch (uTexturePerObjectIdColorsAndFlags, ivec2(3, objectIndex), 0);"); // chipmunk
        
        // get position
        src.push("vec3 position1 = vec3(texelFetch(uTexturePerVertexIdCoordinates, ivec2(indexPositionH.r, indexPositionV.r), 0));")
        src.push("vec3 position2 = vec3(texelFetch(uTexturePerVertexIdCoordinates, ivec2(indexPositionH.g, indexPositionV.g), 0));")
        src.push("vec3 position3 = vec3(texelFetch(uTexturePerVertexIdCoordinates, ivec2(indexPositionH.b, indexPositionV.b), 0));")

        // get normal
        src.push("vec3 normal = -normalize(cross(position3 - position1, position2 - position1));");

        src.push("int vertexNumber = gl_VertexID % 3;");
        src.push("vec3 position;");
        src.push("if (vertexNumber == 0) position = position1;");
        src.push("else if (vertexNumber == 1) position = position2;");
        src.push("else position = position3;");

        // get color
        src.push("uvec4 color = texelFetch (uTexturePerObjectIdColorsAndFlags, ivec2(0, objectIndex), 0);"); // chipmunk

        // flags.x = NOT_RENDERED | COLOR_OPAQUE | COLOR_TRANSPARENT
        // renderPass = COLOR_OPAQUE

        src.push(`if (int(flags.x) != renderPass) {`);
        src.push("   gl_Position = vec4(0.0, 0.0, 0.0, 0.0);"); // Cull vertex

        src.push("} else {");

        src.push("vec4 worldPosition = worldMatrix * (positionsDecodeMatrix * vec4(position, 1.0)); ");
        if (scene.entityOffsetsEnabled) {
            src.push("worldPosition.xyz = worldPosition.xyz + offset;");
        }
        src.push("vec4 viewPosition  = viewMatrix * worldPosition; ");
        src.push("vViewPosition = viewPosition;");
        src.push("vColor = vec4(float(color.r) / 255.0, float(color.g) / 255.0, float(color.b) / 255.0, float(color.a) / 255.0);");

        src.push("vec4 clipPos = projMatrix * viewPosition;");
        if (scene.logarithmicDepthBufferEnabled) {
            src.push("vFragDepth = 1.0 + clipPos.w;");
            src.push("isPerspective = float (isPerspectiveMatrix(projMatrix));");
        }
        if (clipping) {
            src.push("vWorldPosition = worldPosition;");
            src.push("vFlags2 = flags2.r;");
        }
        src.push("gl_Position = clipPos;");
        src.push("}");

        src.push("}");
        return src;
    }

    _buildFragmentShader() {
        const scene = this._scene;
        const lightsState = scene._lightsState;
        const sectionPlanesState = scene._sectionPlanesState;
        const clipping = sectionPlanesState.sectionPlanes.length > 0;
        const src = [];
        src.push ('#version 300 es');
        src.push("// Triangles batching flat-shading draw fragment shader");
        src.push("#extension GL_OES_standard_derivatives : enable");
        src.push("#ifdef GL_FRAGMENT_PRECISION_HIGH");
        src.push("precision highp float;");
        src.push("precision highp int;");
        src.push("#else");
        src.push("precision mediump float;");
        src.push("precision mediump int;");
        src.push("#endif");

        if (scene.logarithmicDepthBufferEnabled) {
            src.push("in float isPerspective;");
            src.push("uniform float logDepthBufFC;");
            src.push("in float vFragDepth;");
        }

        if (this._withSAO) {
            src.push("uniform sampler2D uOcclusionTexture;");
            src.push("uniform vec4      uSAOParams;");

            src.push("const float       packUpscale = 256. / 255.;");
            src.push("const float       unpackDownScale = 255. / 256.;");
            src.push("const vec3        packFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );");
            src.push("const vec4        unPackFactors = unpackDownScale / vec4( packFactors, 1. );");

            src.push("float unpackRGBToFloat( const in vec4 v ) {");
            src.push("    return dot( v, unPackFactors );");
            src.push("}");
        }

        if (clipping) {
            src.push("in vec4 vWorldPosition;");
            src.push("in int vFlags2;");
            for (let i = 0, len = sectionPlanesState.sectionPlanes.length; i < len; i++) {
                src.push("uniform bool sectionPlaneActive" + i + ";");
                src.push("uniform vec3 sectionPlanePos" + i + ";");
                src.push("uniform vec3 sectionPlaneDir" + i + ";");
            }
        }

        src.push("uniform mat4 viewMatrix;");

        src.push("uniform vec4 lightAmbient;");
        for (let i = 0, len = lightsState.lights.length; i < len; i++) {
            const light = lightsState.lights[i];
            if (light.type === "ambient") {
                continue;
            }
            src.push("uniform vec4 lightColor" + i + ";");
            if (light.type === "dir") {
                src.push("uniform vec3 lightDir" + i + ";");
            }
            if (light.type === "point") {
                src.push("uniform vec3 lightPos" + i + ";");
            }
            if (light.type === "spot") {
                src.push("uniform vec3 lightPos" + i + ";");
                src.push("uniform vec3 lightDir" + i + ";");
            }
        }

        src.push("in vec4 vViewPosition;");
        src.push("in vec4 vColor;");
       
        src.push("out vec4 outColor;");
 
        src.push("void main(void) {");

        if (clipping) {
            src.push("  bool clippable = vFlags2 > 0;");
            src.push("  if (clippable) {");
            src.push("  float dist = 0.0;");
            for (let i = 0, len = sectionPlanesState.sectionPlanes.length; i < len; i++) {
                src.push("if (sectionPlaneActive" + i + ") {");
                src.push("   dist += clamp(dot(-sectionPlaneDir" + i + ".xyz, vWorldPosition.xyz - sectionPlanePos" + i + ".xyz), 0.0, 1000.0);");
                src.push("}");
            }
            src.push("  if (dist > 0.0) { ");
            src.push("      discard;")
            src.push("  }");
            src.push("}");
        }

        src.push("vec3 reflectedColor = vec3(0.0, 0.0, 0.0);");
        src.push("vec3 viewLightDir = vec3(0.0, 0.0, -1.0);");

        src.push("float lambertian = 1.0;");

        src.push("vec3 xTangent = dFdx( vViewPosition.xyz );");
        src.push("vec3 yTangent = dFdy( vViewPosition.xyz );");
        src.push("vec3 viewNormal = normalize( cross( xTangent, yTangent ) );");

        for (let i = 0, len = lightsState.lights.length; i < len; i++) {
            const light = lightsState.lights[i];
            if (light.type === "ambient") {
                continue;
            }
            if (light.type === "dir") {
                if (light.space === "view") {
                    src.push("viewLightDir = normalize(lightDir" + i + ");");
                } else {
                    src.push("viewLightDir = normalize((viewMatrix * vec4(lightDir" + i + ", 0.0)).xyz);");
                }
            } else if (light.type === "point") {
                if (light.space === "view") {
                    src.push("viewLightDir = -normalize(lightPos" + i + " - viewPosition.xyz);");
                } else {
                    src.push("viewLightDir = -normalize((viewMatrix * vec4(lightPos" + i + ", 0.0)).xyz);");
                }
            } else if (light.type === "spot") {
                if (light.space === "view") {
                    src.push("viewLightDir = normalize(lightDir" + i + ");");
                } else {
                    src.push("viewLightDir = normalize((viewMatrix * vec4(lightDir" + i + ", 0.0)).xyz);");
                }
            } else {
                continue;
            }

            src.push("lambertian = max(dot(-viewNormal, viewLightDir), 0.0);");
            src.push("reflectedColor += lambertian * (lightColor" + i + ".rgb * lightColor" + i + ".a);");
        }
        
        src.push("vec4 fragColor =  vec4((lightAmbient.rgb * lightAmbient.a * vColor.rgb) + (reflectedColor * vColor.rgb), vColor.a);");

        if (this._withSAO) {
            // Doing SAO blend in the main solid fill draw shader just so that edge lines can be drawn over the top
            // Would be more efficient to defer this, then render lines later, using same depth buffer for Z-reject
            src.push("   float viewportWidth     = uSAOParams[0];");
            src.push("   float viewportHeight    = uSAOParams[1];");
            src.push("   float blendCutoff       = uSAOParams[2];");
            src.push("   float blendFactor       = uSAOParams[3];");
            src.push("   vec2 uv                 = vec2(gl_FragCoord.x / viewportWidth, gl_FragCoord.y / viewportHeight);");
            src.push("   float ambient           = smoothstep(blendCutoff, 1.0, unpackRGBToFloat(texture2D(uOcclusionTexture, uv))) * blendFactor;");
            src.push("   outColor            = vec4(fragColor.rgb * ambient, 1.0);");
        } else {
            src.push("   outColor            = fragColor;");
        }

        if (scene.logarithmicDepthBufferEnabled) {
            src.push("    gl_FragDepth = isPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;");
        }

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

export {TrianglesBatchingFlatColorRenderer};