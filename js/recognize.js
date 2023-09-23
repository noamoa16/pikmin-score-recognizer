import { extractDigitRects } from './number_extractor.js';

export function argmax(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0);
}

/**
 * @param {Float32Array} array
 * @returns {Float32Array}
 */
export function softmax(array){
    let exp_sum = 0;
    for(const value of array){
        exp_sum += Math.exp(value);
    }
    let ret = new Float32Array(array.length);
    for(const [index, value] of array.entries()){
        ret[index] = Math.exp(value) / exp_sum;
    }
    return ret;
}

/**
 * @param {string} onnx_path 
 */
export async function initSession(onnx_path){
    const session = new InferenceSession({ backendHint: 'webgl' });
    await session.loadModel(onnx_path);
    return session;
}

/**
 * 数字のデータからlogitsを計算
 * @param {Float32Array} input 
 * @returns {Promise<Float32Array>}
 */
export async function infer(session, input){
    const inputTensor = new Tensor(input, 'float32', [1, 3, 64, 64]);
    const output = await session.run([inputTensor]);
    const logits = [...output.values()][0].data;
    return logits;
}

/**
 * @param {Uint8Array} imageArray 
 * @param {number} height 
 * @param {number} width 
 * @param {number} channels
 * @param {string} mode - モード(ピクミン2チャレンジ = 2c)
 */
export async function recognize(session, imageArray, height, width, channels, mode){
    const {numberImages: numberImages, numberPositions: numberPositions} = 
        extractDigitRects(imageArray, height, width, channels, mode);
    let logitsList = [];
    for(let i = 0; i < numberImages.length; i++){
        const logits = await infer(session, numberImages[i]);
        logitsList.push(logits);
    }
    return {logitsList: logitsList, numberPositions: numberPositions};
}

/**
 * logitsのリストから推定されたスコアを取得
 * @param {Float32Array[]} logitsList 
 * @returns {number}
 */
export function getAnswerFromLogitsList(logitsList){
    let answerList = [];
    for(let i in logitsList){
        let logits = structuredClone(logitsList[i]);
        // 数字の後に空白は来ない
        if(i > 0 && answerList[i - 1] != ''){
            logits[10] -= 1e6;
        }
        // 最初の桁、または空白の後に0は来ない
        if(i == 0 || (i > 0 && answerList[i - 1] == '')){
            logits[0] -= 1e6;
        }

        let label = argmax(logits);
        if(label == 10){
            label = '';
        }
        answerList.push(label.toString());
    }
    return parseInt(answerList.join(''));
}