#include <iostream>
#include <vector>
#include <random>

#include <omp.h>    // -fopenmp
#include <time.h>
#include <sys/time.h>
#include <chrono>

using namespace std;

int N = 1440;
int BLOCK_SIZE = 80;
int IN_BLOCK = BLOCK_SIZE * BLOCK_SIZE; //кол-во элементов в блоке
int ROW_BLOCKS = N / BLOCK_SIZE;        //колво блоков в строке/столбце

using matrix = vector<vector<float>>;

//заранее посчитанная таблица перевода индексов матрицы в индексы её одномерного построчного блочного представления
vector<vector<int>> inds;

void print(const vector<float>& v);
void print(matrix m);

//случайное вещественное число в диапазоне [min, max]
float getRandomNumber(float min, float max)
{
    float val = rand() % (int)pow(10, 8);
    val = min + (val / pow(10, 8)) * (max - min);
    return val;
}


//нижнетреугольная матрица размера n x n
matrix get_lmatrix(int n)
{
    matrix res(n, vector<float>(n, 0));
    //строки
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < i + 1; ++j)
            res[i][j] = getRandomNumber(0, 10);
    return res;
}


//обычная матрица размера n x n
matrix get_matrix(int n)
{
    matrix res(n, vector<float>(n, 0));
    //строки
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            res[i][j] = getRandomNumber(0, 10);
    return res;
}


//тривиальное умножение матриц за O(n^3)
matrix mul(const matrix& A, const matrix& B)
{
    int n = A.size();
    matrix C(n, vector<float>(n, 0));
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            for(int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}


//номер блока слева направо сверху вниз для построчной записи
int get_block_number_row(int i, int j)
{
    int ii = i/BLOCK_SIZE;
    int jj = j/BLOCK_SIZE;
    return jj + ii * ROW_BLOCKS;
}



//представление матрицы блочными строками
vector<float> get_row_blocks(const matrix& M)
{
    int n = M.size();
    vector<float> res(n * n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
        {
            int k = get_block_number_row(i, j);
            res[k * BLOCK_SIZE * BLOCK_SIZE + j%BLOCK_SIZE + (i%BLOCK_SIZE) * BLOCK_SIZE] = M[i][j];
        }
    return res;
}



//получение индекса в одномерном блочном построчном представлении по i, j - координатам в двумерной матрице
int get_coord_row(int i, int j)
{
    int b = get_block_number_row(i, j);     //номер блока
    int s = b * IN_BLOCK;                   //индекс первого числа в блоке
    s += BLOCK_SIZE * (i % BLOCK_SIZE) + (j % BLOCK_SIZE);
    return s;
}



//А - нижнетреугольная, записана по блочным строкам, В - обычная, записана по блочным строкам
matrix mul_blocks(const vector<float>& A, const vector<float>& B)
{
    int n = sqrt(A.size());
    matrix C(n, vector<float>(n, 0));
    #pragma omp parallel for
    for(int i = 0; i < n; i += BLOCK_SIZE)
        for(int k = 0; k < n; k += BLOCK_SIZE)
            for(int j = 0; j < n; j += BLOCK_SIZE)
                for(int ii = i; (ii < i + BLOCK_SIZE) && (ii < n); ++ii)
                    for(int kk = k; (kk < k + BLOCK_SIZE) && (kk < n); ++kk)
                        for(int jj = j; (jj < j + BLOCK_SIZE) && (jj < n); ++jj)
                            C[ii][jj] += A[inds[ii][kk]] * B[inds[kk][jj]];

    return C;
}


//инициализация таблицы индексов. она универсальна для любой матрицы размера n и её блочного построчного представления
vector<vector<int>> init(int n)
{
    vector<vector<int>> res(n, vector<int>(n));
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            res[i][j] = get_coord_row(i, j);

    return res;
}


//проверка равенства двух матрицы
bool is_equal(const matrix& M1, const matrix& M2)
{
    int n = M1.size();
    double eps = 1e-6;
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            if (fabs(M1[i][j] - M2[i][j]) > eps)
                return false;

    return true;
}

int main() { 
    srand(time(0));
    inds = init(N);
    cout << "N = " <<N<<endl;
    cout << "BLOCK_SIZE = " <<BLOCK_SIZE<<endl;
    //------------------------------------------------------------------------------
    //Последовательно
    matrix A = get_lmatrix(N);
    matrix B = get_matrix(N);
    auto start = chrono::steady_clock::now();
    matrix C1 = mul(A, B);
    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout<<"Последовательно, C1: " <<chrono::duration<double, milli>(diff).count()/1000.0 <<" c."<<endl;
    //print(C1);

    //------------------------------------------------------------------------------
    //Параллельно

    //делители для 2880 (тест)
    //vector<int> sizes2880 = {10, 15, 20, 30, 40, 48, 60, 72, 90, 120, 144, 160, 180, 192, 240, 288, 320, 360, 480, 576, 720, 960, 1440};
    //делители для 3920 (тест)
    //vector<int> sizes3920 = {10, 20, 35, 49, 70, 98, 112, 140, 196, 245, 280, 392, 490, 560, 784, 980, 1960};


    /*for(int size : sizes3920)
    {
        BLOCK_SIZE = size;
        vector<float> A_blocks = get_row_blocks(A);
        vector<float> B_blocks = get_row_blocks(B);
        start = chrono::steady_clock::now();
        matrix C2 = mul_blocks(A_blocks, B_blocks);
        end = chrono::steady_clock::now();
        diff = end - start;
        cout<<"BLOCK_SIZE = "<<BLOCK_SIZE<<endl;
        cout<<"Параллельно, C2: " <<chrono::duration<double, milli>(diff).count()/1000.0 <<" c."<<endl;
    }*/

    vector<float> A_blocks = get_row_blocks(A);
    vector<float> B_blocks = get_row_blocks(B);

    start = chrono::steady_clock::now();
    matrix C2 = mul_blocks(A_blocks, B_blocks);
    end = chrono::steady_clock::now();
    diff = end - start;
    cout<<"Параллельно, C2: " <<chrono::duration<double, milli>(diff).count()/1000.0 <<" c."<<endl;
    //print(C2);


    cout<<"is_equal(C1, C2) = " <<is_equal(C1, C2)<<endl;
    cout<<"end of main()"<<endl;
    return 0;
}



void print(const vector<float>& v)
{
    for (auto i : v)
    {
        cout<<i<<' ';
    }
    cout<<endl;
}

void print(matrix m)
{
    for( auto v : m)
    {
        print(v);
    }
    cout<<endl;
}
