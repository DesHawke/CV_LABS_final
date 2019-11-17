#pragma once
#ifndef IMAGE_FUNCTIONS
#define IMAGE_FUNCTIONS
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct Pixel {
public:
	int x;
	int y;
	double value;
	bool interest;
};


struct PPoint {
public:
	int x;
	int y;
	double value;
};

class Matrix {
public:
	int height;
	int width;
	double** values;

	Matrix();
	Matrix(int, int);
	Matrix(Mat image);
	Matrix& operator=(const Matrix&);
	Matrix(const Matrix& obj);
	void to_zero();
	~Matrix();
};


void lin_norm(Matrix&, int, int);
Mat to_image(Matrix);
void show_image(Mat, string);
void write_image(Mat, string);

int check_edge(int, int, int);
Matrix derivative(Matrix, int(&H)[3][3]);
Matrix sobel_operator(Matrix, Matrix);
Matrix gauss_filter(Matrix, int, double);
Matrix gauss_weight(int, double);
Matrix downsample(Matrix);
int number_of_layers(Matrix);
Matrix* L(Matrix, int, double, double);

double Matrix_trace(Matrix);
double Matrix_det(Matrix);

vector<Pixel> Moravec(Matrix, int, double, double angle = 0);
vector<Pixel> Harris(Matrix, int, double, double angle = 0);

vector<Pixel> ANMS(vector<Pixel>, int);
vector<Pixel> ANMS_rad(vector<Pixel>, int);

Matrix rotate_mtr(Matrix, double);
Mat rotate_img(Mat, double);

void show_result_moravec(string, int, double, int, double angle = 0);
void show_result_harris(string, int, double, int, double angle = 0);

bool find_mx(Matrix F, int sizeW, int x, int y);
double gradients(Matrix);
#endif