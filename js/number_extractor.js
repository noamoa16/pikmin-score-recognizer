/**
 * 画像を整数倍に拡大する
 * @param {nj.NdArray} imageNj 
 * @param {number} times - 拡大率(整数)
 * @returns {nj.NdArray}
 */
function magnify(imageNj, times){
    if(times == 1){
        return imageNj;
    }
    const H = imageNj.shape[0];
    const W = imageNj.shape[1];
    const C = imageNj.shape[2];
    let newImageNj = nj.zeros([H * times, W * times, C]);
    for(let h = 0; h < H * times; h++){
        for(let w = 0; w < W * times; w++){
            for(let c = 0; c < C; c++){
                newImageNj.set(h, w, c, imageNj.get(Math.floor(h / times),Math.floor(w / times), c));
            }
        }
    }
    return newImageNj;
}

/**
 * @param {Uint8Array} imageArray 
 * @param {number} height
 * @param {number} width
 * @param {number} channels
 * @param {string} mode - モード(ピクミン2チャレンジ = 2c)
 */
export function extractDigitRects(imageArray, height, width, channels, mode){
    const MODE_TO_NUM_DIGITS = {
        '2c': 5,
    };
    const numDigits = MODE_TO_NUM_DIGITS[mode];

    let imageNj = nj.array(Array.from(imageArray));
    imageNj = imageNj.reshape(height, width, channels);
    imageNj = imageNj.slice(null, null, [null, 3]); // アルファチャネルを削除

    const SIZE_OF_I32 = 4;
    const extract = Module.cwrap(
        `extract_digit_rects_emscripten_${mode}`, 
        null, 
        ['array', 'number', 'number', 'number', 'number']
    );
    const array = new Uint8Array(imageNj.flatten().tolist());
    let buffer = Module._malloc(numDigits * 4 * SIZE_OF_I32);
    extract(array, height, width, 3, buffer);
    /** @type {number[]} */
    let result = [];
    for(let i = 0; i < numDigits * 4; i++){
        result.push(Module.getValue(buffer + i * SIZE_OF_I32, 'i32'))
    }
    Module._free(buffer);

    // 抽出結果を配列として格納
    let numberImages = [];
    let numberPositions = [];
    for(let i = 0; i < numDigits; i++){
        const y = result[i * 4];
        const x = result[i * 4 + 1];
        const h = result[i * 4 + 2];
        const w = result[i * 4 + 3];
        numberPositions.push({x: x, y: y, w: w, h: h});
        const array = njArrayToFloat32Array(imageNj.slice(
            [y, y + h],
            [x, x + w],
            null
        ));
        numberImages.push(array);
    }
    
    /**
     * @param {nj.NdArray} njArray
     * @returns {Float32Array}
     */
    function njArrayToFloat32Array(njArray){
        const magnified = magnify(njArray, Math.ceil(64 / Math.min(njArray.shape[0], njArray.shape[1])));
        const njArray64x64 = nj.array(nj.images.resize(magnified, 64, 64).slice(null, null, [null, 3]).tolist(), 'array');
        return new Float32Array(njArray64x64.transpose(2, 0, 1).divide(255.0).flatten().tolist());
    }

    return {numberImages: numberImages, numberPositions: numberPositions};
}