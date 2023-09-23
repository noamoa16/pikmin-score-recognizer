#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>

constexpr int DIR8_VECTORS[8][2] = {
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1},
    {-1, 0},
    {-1, -1},
    {0, -1},
    {1, -1}
};

struct PointInt{
    int x;
    int y;
    PointInt(const int x, const int y){
        this->x = x;
        this->y = y;
    }
    PointInt operator+(const PointInt &rhs) const{
        return {
            this->x + rhs.x,
            this->y + rhs.y
        };
    }
    PointInt operator-(const PointInt &rhs) const{
        return {
            this->x - rhs.x,
            this->y - rhs.y
        };
    }
};

bool get_value(const std::vector<std::vector<bool>> &mat, const int i, const int j){
    if(0 <= i && i < (int)mat.size() && 0 <= j && j < (int)mat[0].size()){
        return mat[i][j];
    }
    else{
        return false;
    }
}

std::vector<std::vector<bool>> erode3x3(const std::vector<std::vector<bool>> &mat){
    const int H = mat.size(), W = mat[0].size();
    std::vector<std::vector<bool>> ret(H, std::vector<bool>(W, false));
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            bool value = mat[i][j];
            for(int k = 0; k < 8; k++){
                const int dx = DIR8_VECTORS[k][0], dy = DIR8_VECTORS[k][1];
                value &= get_value(mat, i + dx, j + dy);
            }
            ret[i][j] = value;
        }
    }
    return ret;
}

std::vector<std::vector<bool>> dilate3x3(const std::vector<std::vector<bool>> &mat){
    const int H = mat.size(), W = mat[0].size();
    std::vector<std::vector<bool>> ret(H, std::vector<bool>(W, false));
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            bool value = mat[i][j];
            for(int k = 0; k < 8; k++){
                const int dx = DIR8_VECTORS[k][0], dy = DIR8_VECTORS[k][1];
                value |= get_value(mat, i + dx, j + dy);
            }
            ret[i][j] = value;
        }
    }
    return ret;
}

std::vector<float> conv3(const std::vector<float> &v){
    const int n = v.size();
    std::vector<float> ret(n);
    ret[0] = (v[0] + v[1]) / 3;
    for(int i = 1; i < n - 1; i++){
        ret[i] = (v[i - 1] + v[i] + v[i + 1]) / 3;
    }
    ret[n - 1] = (v[n - 2] + v[n - 1]) / 3;
    return ret;
}

template <typename T>
std::vector<int> extract_digit_rects_2c(
        const std::vector<T> &array, 
        const int height, 
        const int width, 
        const int channels
    ){
    constexpr int PIKMIN2_RESULT_RED[] = {222, 13, 9};
        
    std::vector<std::vector<bool>> red_area(height, std::vector<bool>(width, false));
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            const int index = (j + i * width) * channels;
            int squared_dist = 0;
            for(int k = 0; k < 3; k++){
                const int diff = array[index + k] - PIKMIN2_RESULT_RED[k];
                squared_dist += diff * diff;
            }
            red_area[i][j] = 
                squared_dist / (255.0 * 255.0) <= 0.5 * 0.5;
        }
    }

    red_area = dilate3x3(red_area);
    red_area = erode3x3(red_area);
    red_area = erode3x3(red_area);
    red_area = dilate3x3(red_area);

    const int min_count = width / 30;

    // 赤いピクセル数をカウント
    std::vector<float> count(height);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            count[i] += red_area[i][j] ? 1 : 0;
        }
    }
    count = conv3(count);

    // ハイスコアの座標を求める
    int highscore_top = -1, highscore_bottom = -1;
    for(int i = height - 1; i >= 0; i--){
        if(count[i] >= min_count){
            if(highscore_bottom == -1){
                highscore_bottom = i;
            }
            highscore_top = i;
        }
        else if(highscore_bottom != -1){
            break;
        }
    }
    
    std::vector<int> first(highscore_bottom - highscore_top + 1);
    for(int i = 0; i < highscore_bottom - highscore_top + 1; i++){
        for(int j = 0; j < width; j++){
            if(red_area[i + highscore_top][j]){
                break;
            }
            first[i] = j;
        }
    }
    std::vector<int> last(highscore_bottom - highscore_top + 1);
    for(int i = 0; i < highscore_bottom - highscore_top + 1; i++){
        for(int j = width - 1; j >= 0; j--){
            if(red_area[i + highscore_top][j]){
                break;
            }
            last[i] = j;
        }
    }
    const int gap = (int)((highscore_bottom - highscore_top) * 0.25);
    std::sort(first.begin(), first.end());
    std::sort(last.begin(), last.end());
    const int highscore_left = first[gap];
    const int highscore_right = last[highscore_bottom - highscore_top - gap];

    std::vector<float> narrow_count(height);
    for(int i = 0; i < height; i++){
        for(int j = highscore_left; j <= highscore_right; j++){
            narrow_count[i] += red_area[i][j] ? 1 : 0;
        }
    }
    narrow_count = conv3(narrow_count);

    // 残りピクミンの座標を求める
    int remaining_pikmin_top = -1, remaining_pikmin_bottom = -1;
    for(int i = 2 * highscore_top - highscore_bottom - 1; i >= 0; i--){
        if(narrow_count[i] >= min_count){
            if(remaining_pikmin_bottom == -1){
                remaining_pikmin_bottom = i;
            }
            remaining_pikmin_top = i;
        }
        else if(remaining_pikmin_bottom != -1){
            break;
        }
    }

    const PointInt highscore_pos = {
        (highscore_left + highscore_right) / 2,
        (highscore_top + highscore_bottom) / 2
    };
    const PointInt remaining_pikmin_pos = {
        highscore_pos.x, 
        (int)((remaining_pikmin_top + remaining_pikmin_bottom) / 2)
    };
    const int length = highscore_pos.y - remaining_pikmin_pos.y; // ハイスコアと残りピクミンの距離
    const bool is_wide = ((float)highscore_right - highscore_left) / length >= 2.55; // 2.42 ~ 2.68
    const int digits_size = is_wide ? (int)(0.5 * length) : (int)(0.475 * length);
    const float digits_stride = is_wide ? 1.15 * digits_size : digits_size;
    const PointInt digits_start_pos = remaining_pikmin_pos - PointInt(
        (int)(2.0 * digits_stride + digits_size / 2.0),
        (int)(1.65 * digits_size)
    );

    // スコアの座標を四角形として計算
    std::vector<int> result(20);
    for(int i = 0; i < 5; i++){
        const PointInt rect_start = {
            digits_start_pos.x + (int)(i * digits_stride), 
            digits_start_pos.y
        };
        const PointInt rect_end = {
            digits_start_pos.x + digits_size + (int)(i * digits_stride),
            digits_start_pos.y + digits_size
        };
        const PointInt size = rect_end - rect_start;
        result[i * 4] = rect_start.y;
        result[i * 4 + 1] = rect_start.x;
        result[i * 4 + 2] = size.y;
        result[i * 4 + 3] = size.x;
    }

    return result;
}


extern "C" {
    #ifdef EMSCRIPTEN
    EMSCRIPTEN_KEEPALIVE
    #endif
    void extract_digit_rects_emscripten_2c(
            const uint8_t *array, 
            const int height, 
            const int width, 
            const int channels,
            int *result
        ){
        const std::vector<uint8_t> vector(array, array + height * width * channels);
        const std::vector<int> result_vector = extract_digit_rects_2c<uint8_t>(vector, height, width, channels);
        for(int i = 0; i < 20; i++){
            result[i] = result_vector[i];
        }
    }
}