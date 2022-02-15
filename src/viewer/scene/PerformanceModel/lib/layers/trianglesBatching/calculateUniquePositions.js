/**
 * @author https://github.com/tmarti
 * @license MIT
 * 
 * This file takes a geometry given by { positions, indices }, and returns
 * equivalent { positions, indices } arrays but which only contain unique
 * positions.
 * 
 * The time is O(N logN) with the number of positions due to a pre-sorting
 * step, but is much more GC-friendly and actually faster than the classic O(N)
 * approach based in keeping a hash-based LUT to identify unique positions.
 */
 let comparePositions = null;

function compareVertex (a, b) {
    let res;

    for (let i = 0; i < 3; i++) {
        if (0!= (res = comparePositions[a*3+i] - comparePositions[b*3+i]))
        {
            return res;
        }
    }

    return 0;
};

let seqInit = null;

function setMaxNumberOfPositions (maxPositions)
{
    if (seqInit !== null && seqInit.length >= maxPositions)
    {
        return;
    }

    seqInit = new Uint32Array(maxPositions);

    for (let i = 0; i < maxPositions; i++)
    {
        seqInit[i] = i;
    }
}

function uniquifyPositions(mesh)
{
    let _positions = mesh.positions;
    let _indices = mesh.indices;

    setMaxNumberOfPositions(_positions.length / 3);

    let seq = seqInit.slice (0, _positions.length / 3);
    let remappings = seqInit.slice (0, _positions.length / 3);

    comparePositions = _positions;

    seq.sort(compareVertex);

    let uniqueIdx = 0

    remappings[seq[0]] = 0;

    for (let i = 1, len = seq.length; i < len; i++)
    {
        if (0 != compareVertex(seq[i], seq[i-1]))
        {
            uniqueIdx++;
        }

        remappings[seq[i]] = uniqueIdx;
    }

    const numUniquePositions = uniqueIdx + 1;

    const newPositions = new Uint16Array (numUniquePositions * 3);

    uniqueIdx = 0

    newPositions [uniqueIdx * 3 + 0] = _positions [seq[0] * 3 + 0];
    newPositions [uniqueIdx * 3 + 1] = _positions [seq[0] * 3 + 1];
    newPositions [uniqueIdx * 3 + 2] = _positions [seq[0] * 3 + 2];
    
    for (let i = 1, len = seq.length; i < len; i++)
    {
        if (0 != compareVertex(seq[i], seq[i-1]))
        {
            uniqueIdx++;

            newPositions [uniqueIdx * 3 + 0] = _positions [seq[i] * 3 + 0];
            newPositions [uniqueIdx * 3 + 1] = _positions [seq[i] * 3 + 1];
            newPositions [uniqueIdx * 3 + 2] = _positions [seq[i] * 3 + 2];
        }

        remappings[seq[i]] = uniqueIdx;
    }

    comparePositions = null;

    let newIndices = new Uint32Array (_indices.length);

    for (let i = 0, len = _indices.length; i < len; i++)
    {
        newIndices[i] = remappings [
            _indices[i]
        ];
    }

    return [
        newPositions,
        newIndices
    ];
}


export { uniquifyPositions }