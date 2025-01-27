import {clusterizeV2} from "./cluster-helper";
import {math} from "../math";

const tempVec3 = math.vec3();

/**
 * Number of bits per-dimension in the 2-dimensional LUT fast atan table
 */
const ATAN2_LUT_BITS = 9;
const ATAN2_FACTOR = 1 << (ATAN2_LUT_BITS - 1);

/**
 * Constant for quick conversion of radians to degrees
 */
const _180_DIV_MATH_PI = 180 / Math.PI;
const atan2LUT = new Float32Array((1 << ATAN2_LUT_BITS) * (1 << ATAN2_LUT_BITS));

// Initialize the Look Up Table
for (let i = -ATAN2_FACTOR; i < ATAN2_FACTOR; i++) {
    for (let j = -ATAN2_FACTOR; j < ATAN2_FACTOR; j++) {
        const index = ((i + ATAN2_FACTOR) << ATAN2_LUT_BITS) + (j + ATAN2_FACTOR);
        const max = Math.max(Math.abs(i), Math.abs(j));
        atan2LUT [index] = Math.atan2(i / max, j / max);
    }
}

/**
 * Fast ````Math.atan2```` implementation based in Look Up Tables.
 *
 * @param {number} x
 * @param {number} y
 *
 * @returns {number}
 */
function fastAtan2(x, y) {
    const max_factor = ATAN2_FACTOR / Math.max(Math.abs(x), Math.abs(y));
    const xx = Math.round(x * max_factor) + (ATAN2_FACTOR - 1);
    const yy = Math.round(y * max_factor) + (ATAN2_FACTOR - 1);
    return atan2LUT [(xx << ATAN2_LUT_BITS) + yy];
}

const VISIBILITY_CHECK_ALL_D = (1 << 0);
const VISIBILITY_CHECK_NONE_D = (1 << 1);
const VISIBILITY_CHECK_D_LESS = (1 << 2);
const VISIBILITY_CHECK_D_MORE = (1 << 3);

const VISIBILITY_CHECK_ALL_H = (1 << 4);
const VISIBILITY_CHECK_NONE_H = (1 << 5);
const VISIBILITY_CHECK_H_LESS = (1 << 6);
const VISIBILITY_CHECK_H_MORE = (1 << 7);

const VISIBILITY_CHECK_ALL_V = (1 << 8);
const VISIBILITY_CHECK_NONE_V = (1 << 9);
const VISIBILITY_CHECK_V_LESS = (1 << 10);
const VISIBILITY_CHECK_V_MORE = (1 << 11);

const VISIBILITY_CHECK_ENVOLVES_D = (1 << 12);
const VISIBILITY_CHECK_ENVOLVES_H = (1 << 13);
const VISIBILITY_CHECK_ENVOLVES_V = (1 << 14);

/**
 * Data structure containing pre-initialized `View Frustum Culling` data.
 *
 * Will be used by the rest of `View Frustum Culling` related code.
 *
 * @private
 */
export class VFCState {

    constructor() {

        this._aabbTree = null;
        this._orderedMeshList = [];
        this._orderedEntityList = [];
        this._frustumProps = {
            dirty: true,
            wMultiply: 1.0,
            hMultiply: 1.0,
        };
        this._cullFrame = 0;

        /**
         * @type {boolean}
         * @private
         */
        this.finalized = false;
    }

    /**
     *
     * @param {Array<object>} entityList
     * @param {Array<object>} meshList
     */
    initializeVFCState(entityList, meshList) {
        if (this.finalized) {
            throw "Already finalized";
        }
        const clusterResult = clusterizeV2(entityList, meshList);
        this._aabbTree = clusterResult.rTreeBasedAabbTree;
        for (let i = 0, len = clusterResult.orderedClusteredIndexes.length; i < len; i++) {
            const entityIndex = clusterResult.orderedClusteredIndexes[i];
            const clusterNumber = clusterResult.entityIdToClusterIdMapping[entityIndex];
            const entity = entityList[entityIndex];
            const newMeshIds = [];
            for (let j = 0, len2 = entity.meshIds.length; j < len2; j++) {
                const meshIndex = entity.meshIds[j];
                meshList[meshIndex].id = this._orderedMeshList.length;
                newMeshIds.push(this._orderedMeshList.length);
                this._orderedMeshList.push({
                    clusterNumber: clusterNumber,
                    mesh: meshList[meshIndex]
                });
            }
            entity.meshIds = newMeshIds;
            this._orderedEntityList.push(entity);
        }
        for (let i = 0, len = clusterResult.instancedIndexes.length; i < len; i++) {
            const entityIndex = clusterResult.instancedIndexes[i];
            let entity = entityList[entityIndex];
            const newMeshIds = [];
            for (let j = 0, len2 = entity.meshIds.length; j < len2; j++) {
                const meshIndex = entity.meshIds[j];
                meshList[meshIndex].id = this._orderedMeshList.length;
                newMeshIds.push(this._orderedMeshList.length);
                this._orderedMeshList.push({clusterNumber: 99999, mesh: meshList[meshIndex]});
            }
            entity.meshIds = newMeshIds;
            this._orderedEntityList.push(entity);
        }
    }

    /**
     * @param {SceneModel} sceneModel
     * @param {*} fnForceFinalizeLayer
     */
    finalize(sceneModel, fnForceFinalizeLayer) {
        if (this.finalized) {
            throw "Already finalized";
        }
        let lastClusterNumber = -1;
        for (let i = 0, len = this._orderedMeshList.length; i < len; i++) {
            const {clusterNumber, mesh} = this._orderedMeshList [i];
            if (lastClusterNumber !== -1 && lastClusterNumber !== clusterNumber) {
                fnForceFinalizeLayer.call(sceneModel);
            }
            sceneModel._createMesh(mesh);
            lastClusterNumber = clusterNumber;
        }
        // fnForceFinalizeLayer ();
        for (let i = 0, len = this._orderedEntityList.length; i < len; i++) {
            sceneModel._createEntity(this._orderedEntityList[i])
        }
        // Free memory
        this._orderedMeshList = [];
        this._orderedEntityList = [];
        this.finalized = true;
    }

    /**
     * @param {SceneModel} sceneModel
     */
    applyViewFrustumCulling(sceneModel) {
        if (!this.finalized) {
            throw "Not finalized";
        }
        if (!this._aabbTree) {
            return;
        }
        if (!this._canvasElement) {
            this._canvasElement = sceneModel.scene.canvas.canvas;
        }
        if (!this._camera) {
            this._camera = sceneModel.scene.camera;
        }
        this._ensureFrustumPropsUpdated(sceneModel);
        this._initializeCullingDataIfNeeded(sceneModel);
        const visibleNodes = this._searchVisibleNodesWithFrustumCulling();
        this._cullFrame++;
        this._markVisibleFrameOfVisibleNodes(visibleNodes, this._cullFrame);
        this._cullNonVisibleNodes(sceneModel, this._cullFrame);
    }

    _initializeCullingDataIfNeeded(sceneModel) {
        if (this._internalNodesList) {
            return;
        }
        if (!this._aabbTree) {
            return;
        }
        const allAabbNodes = this._aabbTree.all();
        let maxEntityId = 0;
        allAabbNodes.forEach(aabbbNode => {
            maxEntityId = Math.max(maxEntityId, aabbbNode.entity.id)
        });
        const internalNodesList = new Array(maxEntityId + 1);
        allAabbNodes.forEach(aabbbNode => {
            internalNodesList [aabbbNode.entity.id] = sceneModel.objects[aabbbNode.entity.xeokitId];
        });
        this._internalNodesList = internalNodesList;
        this._lastVisibleFrameOfNodes = new Array(internalNodesList.length);
        this._lastVisibleFrameOfNodes.fill(0);
    }

    _searchVisibleNodesWithFrustumCulling() {
        return this._aabbTree.searchCustom((bbox, isLeaf) => this._aabbIntersectsCameraFrustum(bbox, isLeaf), (bbox) => this._aabbContainedInCameraFrustum(bbox))
    }

    _markVisibleFrameOfVisibleNodes(visibleNodes, cullFrame) {
        const lastVisibleFrameOfNodes = this._lastVisibleFrameOfNodes;
        for (let i = 0, len = visibleNodes.length; i < len; i++) {
            lastVisibleFrameOfNodes [visibleNodes[i].entity.id] = cullFrame;
        }
    }

    _cullNonVisibleNodes(sceneModel, cullFrame) {
        const internalNodesList = this._internalNodesList;
        const lastVisibleFrameOfNodes = this._lastVisibleFrameOfNodes;
        for (let i = 0, len = internalNodesList.length; i < len; i++) {
            if (internalNodesList[i]) {
                internalNodesList[i].culledVFC = lastVisibleFrameOfNodes[i] !== cullFrame;
            }
        }
    }

    /**
     * Returns all 8 coordinates of an AABB.
     *
     * @param {Array<number>} bbox An AABB
     *
     * @private
     */
    _getPointsForBBox(bbox) {
        const points = [];
        for (let i = 0; i < 8; i++) {
            points.push(new Float32Array([(i & 1) ? bbox.maxX : bbox.minX, (i & 2) ? bbox.maxY : bbox.minY, (i & 4) ? bbox.maxZ : bbox.minZ]));
        }
        return points;
    }

    /**
     * @param {*} bbox
     * @param {*} isLeaf
     * @returns
     *
     * @private
     */
    _aabbIntersectsCameraFrustum(bbox, isLeaf) {
        if (isLeaf) {
            return true;
        }
        if (this._camera.projection === "ortho") {          // TODO: manage ortho views
            this._frustumProps.dirty = false;
            return true;
        }
        // numIntersectionChecks++;
        const check = this._aabbIntersectsCameraFrustum_internal(bbox);
        const interD = !(check & VISIBILITY_CHECK_ALL_D) && !(check & VISIBILITY_CHECK_NONE_D);
        const interH = !(check & VISIBILITY_CHECK_ALL_H) && !(check & VISIBILITY_CHECK_NONE_H);
        const interV = !(check & VISIBILITY_CHECK_ALL_V) && !(check & VISIBILITY_CHECK_NONE_V);
        if (((check & VISIBILITY_CHECK_ENVOLVES_D) || interD || (check & VISIBILITY_CHECK_ALL_D)) &&
            ((check & VISIBILITY_CHECK_ENVOLVES_H) || interH || (check & VISIBILITY_CHECK_ALL_H)) &&
            ((check & VISIBILITY_CHECK_ENVOLVES_V) || interV || (check & VISIBILITY_CHECK_ALL_V))) {
            return true;
        }
        return false;
    }

    /**
     * @param {*} bbox
     * @returns
     *
     * @private
     */
    _aabbContainedInCameraFrustum(bbox) {
        if (this._camera.projection === "ortho") {    // TODO: manage ortho views
            this._frustumProps.dirty = false;
            return true;
        }
        const check = bbox._check;
        return (check & VISIBILITY_CHECK_ALL_D) && (check & VISIBILITY_CHECK_ALL_H) && (check & VISIBILITY_CHECK_ALL_V);
    }

    /**
     * @param {SceneModel} sceneModel
     *
     * @private
     */
    _ensureFrustumPropsUpdated(sceneModel) {
        const min = Math.min(this._canvasElement.width, this._canvasElement.height); // Assuming "min" for fovAxis
        this._frustumProps.wMultiply = this._canvasElement.width / min;
        this._frustumProps.hMultiply = this._canvasElement.height / min;
        const aspect = this._canvasElement.width / this._canvasElement.height;
        let fov = this._camera.perspective.fov;
        if (aspect < 1) {
            fov = fov / aspect;
        }
        fov = Math.min(fov, 120);
        this._frustumProps.fov = fov;
        // if (!this._frustumProps.dirty)
        // {
        //     return;
        // }
        // Adjust camera eye/look to take into account the `sceneModel.worldMatrix`:
        //  - the entities' AABBs don't take it into account
        //  - and they can't, since `sceneModel.worldMatrix` is dynamic
        // So, instead of transformating the positions of the r*tree's AABBs,
        // apply the inverse transform to the camera eye/look, since the culling
        // result is equivalent.
        const invWorldMatrix = math.inverseMat4(sceneModel.worldMatrix, math.mat4());
        const modelCamEye = math.transformVec3(invWorldMatrix, this._camera.eye, [0, 0, 0]);
        const modelCamLook = math.transformVec3(invWorldMatrix, this._camera.look, [0, 0, 0]);
        this._frustumProps.forward = math.normalizeVec3(math.subVec3(modelCamLook, modelCamEye, [0, 0, 0]), [0, 0, 0]);
        this._frustumProps.up = math.normalizeVec3(this._camera.up, [0, 0, 0]);
        this._frustumProps.right = math.normalizeVec3(math.cross3Vec3(this._frustumProps.forward, this._frustumProps.up, [0, 0, 0]), [0, 0, 0]);
        this._frustumProps.eye = modelCamEye.slice();
        this._frustumProps.CAM_FACTOR_1 = this._frustumProps.fov / 2 * this._frustumProps.wMultiply / _180_DIV_MATH_PI;
        this._frustumProps.CAM_FACTOR_2 = this._frustumProps.fov / 2 * this._frustumProps.hMultiply / _180_DIV_MATH_PI;
        // this._frustumProps.dirty = false;
    }

    /**
     * @param {*} bbox
     * @returns
     *
     * @private
     */
    _aabbIntersectsCameraFrustum_internal(bbox) {
        const bboxPoints = bbox._points || this._getPointsForBBox(bbox);
        bbox._points = bboxPoints;
        let retVal =
            VISIBILITY_CHECK_ALL_D | VISIBILITY_CHECK_NONE_D |
            VISIBILITY_CHECK_ALL_H | VISIBILITY_CHECK_NONE_H |
            VISIBILITY_CHECK_ALL_V | VISIBILITY_CHECK_NONE_V;
        for (let i = 0, len = bboxPoints.length; i < len; i++) {
            // if ((!(retVal & VISIBILITY_CHECK_ALL_D) && !(retVal & VISIBILITY_CHECK_NONE_D)) ||
            //     (!(retVal & VISIBILITY_CHECK_ALL_H) && !(retVal & VISIBILITY_CHECK_NONE_H)) ||
            //     (!(retVal & VISIBILITY_CHECK_ALL_V) && !(retVal & VISIBILITY_CHECK_NONE_V)))
            // {
            //     break;
            // }
            const bboxPoint = bboxPoints [i];
            const pointRelToCam = tempVec3;
            pointRelToCam[0] = bboxPoint[0] - this._frustumProps.eye[0];
            pointRelToCam[1] = bboxPoint[1] - this._frustumProps.eye[1];
            pointRelToCam[2] = bboxPoint[2] - this._frustumProps.eye[2];
            const forwardComponent = math.dotVec3(pointRelToCam, this._frustumProps.forward);
            if (forwardComponent < 0) {
                retVal |= VISIBILITY_CHECK_D_LESS;
                retVal &= ~VISIBILITY_CHECK_ALL_D;
            } else {
                retVal |= VISIBILITY_CHECK_D_MORE;
                retVal &= ~VISIBILITY_CHECK_NONE_D;
            }
            const rightComponent = math.dotVec3(pointRelToCam, this._frustumProps.right);
            const rightAngle = fastAtan2(rightComponent, forwardComponent);
            if (Math.abs(rightAngle) > this._frustumProps.CAM_FACTOR_1) {
                if (rightAngle < 0) {
                    retVal |= VISIBILITY_CHECK_H_LESS;
                } else {
                    retVal |= VISIBILITY_CHECK_H_MORE;
                }
                retVal &= ~VISIBILITY_CHECK_ALL_H;
            } else {
                retVal &= ~VISIBILITY_CHECK_NONE_H;
            }
            const upComponent = math.dotVec3(pointRelToCam, this._frustumProps.up);
            const upAngle = fastAtan2(upComponent, forwardComponent);
            if (Math.abs(upAngle) > this._frustumProps.CAM_FACTOR_2) {
                if (upAngle < 0) {
                    retVal |= VISIBILITY_CHECK_V_LESS;
                } else {
                    retVal |= VISIBILITY_CHECK_V_MORE;
                }
                retVal &= ~VISIBILITY_CHECK_ALL_V;
            } else {
                retVal &= ~VISIBILITY_CHECK_NONE_V;
            }
        }
        if ((retVal & VISIBILITY_CHECK_D_LESS) && (retVal & VISIBILITY_CHECK_D_MORE)) {
            retVal |= VISIBILITY_CHECK_ENVOLVES_D;
        }
        if ((retVal & VISIBILITY_CHECK_H_LESS) && (retVal & VISIBILITY_CHECK_H_MORE)) {
            retVal |= VISIBILITY_CHECK_ENVOLVES_H;
        }
        if ((retVal & VISIBILITY_CHECK_V_LESS) && (retVal & VISIBILITY_CHECK_V_MORE)) {
            retVal |= VISIBILITY_CHECK_ENVOLVES_V;
        }
        bbox._check = retVal;
        return retVal;
    }
}