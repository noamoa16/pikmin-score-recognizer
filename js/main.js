import {
    initSession,
    recognize, 
    getAnswerFromLogitsList, 
    argmax, 
    softmax,
} from './recognize.js';

/**
 * logitsの配列から、確率表のHTML文字列を作成
 * @param {Float32Array[]} logitsList 
 * @returns {string}
 */
function logitsToTable(logitsList){
    /** @type {number[][]} */
    let probMatrix = [];
    for(const logits of logitsList){
        const probs = Array.from(softmax(logits));
        probMatrix.push(probs);
    }

   let logitsTable = '<table border="1">\n<thead>\n<tr>\n<th></th>\n';
   logitsTable += [...Array(10).keys(), '空白'].map(x => `<td>${x}</td>\n`).join('');
   logitsTable += '</tr>\n</thead>\n<tbody>\n';
   for(let i = 0; i < logitsList.length; i++){
       const ans = argmax(logitsList[i]);
       logitsTable += '<tr>\n';
       logitsTable += `<td>${i + 1}番目の数字</td>\n`;
       for(let j = 0; j < probMatrix[i].length; j++){
           const bTags = [ans == j ? '<b>' : '', ans == j ? '</b>' : ''];
           logitsTable += `<td>${bTags[0]}${(probMatrix[i][j] * 100).toFixed(2)}%${bTags[1]}</td>\n`;
       }
       logitsTable += '</tr>\n';
   }
   logitsTable += '</tbody>\n</table>';
   return logitsTable
}

async function main(){
    // 初期化
    const session = await initSession(ONNX_PATH);

    $(() => {

        // ファイルが選択されたとき
        $('#imageFileInput').on('change', e => {
            /** @type {File} */
            const file = e.target.files[0];

            // ファイルが空でなければ
            if(file){
                const startTime = performance.now();
                $('#result').html('処理中…');
                
                // 画面に画像を表示
                const CANVAS_HEIGHT = 360;
                /** @type {HTMLCanvasElement} */
                const canvas = document.getElementById('imageCanvas');
                /** @type {CanvasRenderingContext2D} */
                const context = canvas.getContext('2d');
                const reader = new FileReader();
                let imageSize = {x: -1, y: -1};
                reader.onload = function(e) {
                    let image = new Image();
                    image.src = e.target.result;
                    image.onload = function() {
                        const extRate = CANVAS_HEIGHT / image.height;
                        canvas.width = image.width * extRate;
                        canvas.height = image.height * extRate;
                        context.drawImage(image, 0, 0, canvas.width, canvas.height);
                        imageSize.x = image.width;
                        imageSize.y = image.height;
                    };
                };
                reader.readAsDataURL(file);
                
                const url = URL.createObjectURL(file);

                // 画像読み込み
                Jimp.read(url).then(image => {
                    const imageLoadEndTime = performance.now();
                    /** @type {Uint8Array} */
                    const imageArray = image.bitmap.data; // 縦 * 横 * RGBA
                    /** @type {number} */
                    const height = image.bitmap.height;
                    /** @type {number} */
                    const width = image.bitmap.width;
                    const channels = 4;
                    const mode = '2c'; // 現在はピクミン2チャレンジモードにのみ対応

                    // スコア認識
                    recognize(session, imageArray, height, width, channels, mode).then(result => {
                        const logitsList = result.logitsList;
                        const numberPositions = result.numberPositions;

                        // スコアを求める
                        const answer = getAnswerFromLogitsList(logitsList);

                        // スコアの位置に四角形描画
                        const extRate = canvas.height / imageSize.y;
                        context.strokeStyle = 'red';
                        context.lineWidth = Math.max(
                            Math.floor(Math.sqrt(height * width) / 2000),
                            1,
                        );
                        for(let i = 0; i < logitsList.length; i++){
                            context.strokeRect(
                                numberPositions[i].x * extRate,
                                numberPositions[i].y * extRate,
                                numberPositions[i].w * extRate,
                                numberPositions[i].h * extRate
                            );
                        }

                        // 結果表示
                        const endTime = performance.now();
                        const logitsTable = logitsToTable(logitsList);
                        const resultText = [
                            `認識結果: ${answer}`,
                            `画像サイズ: ${imageSize.x} x ${imageSize.y}`,
                            `処理時間: ${(endTime - startTime) / 1000} 秒 (画像ロード: ${(imageLoadEndTime - startTime) / 1000} 秒、スコア検出: ${(endTime - imageLoadEndTime) / 1000} 秒)`,
                            logitsTable,
                        ].join('<br>');
                        $('#result').html(resultText);
                    })
                    .catch(e => {
                        $('#result').html(`スコア認識に失敗しました<br>${e}`);
                        console.log(e);
                    });
                });
            }
        });
    });
}

main();