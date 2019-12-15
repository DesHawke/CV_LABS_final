#pragma once
#ifndef IMAGE_FUNCTIONS
#define IMAGE_FUNCTIONS
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

struct Pixel {
public:
	int x;
	int y;
	double value;
	bool interest;
	//double phi;
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


class Descriptor
{
public:
	Pixel interPoint;    // Интересная точка - центр
	int N;
	double *data; // Количество корзин * кол-во гистограмм

	Descriptor() { }

	Descriptor(int size, Pixel point)
	{
		N = size;
		data = new double[size];
		for (int x = 0; x < size; x++) {
			data[x] = 0.0;
		}
		interPoint = point;
	}

	void normalize();
	void clampData(double min, double max);
};

double Clamp(double min, double max, double value);

vector<Descriptor> getDescriptors(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int barCharCount);
vector<Descriptor> getDescriptorsInvRot(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int barCharCount);
vector<Descriptor> getDescriptorsInvRot2(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int barCharCount);

vector<Descriptor> GET_NEW_DescriptorsInvRot(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int barCharCount);
vector<Descriptor> GET_NEW_DescriptorsInvRot2(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int barCharCount);
vector<double> GET_NEW_PointOrientation(Matrix image_dx, Matrix image_dy, Pixel point, int radius);

vector<double> getPointOrientation(Matrix image_dx, Matrix image_dy, Pixel point, int radius);
/* Поиск пика */
double getPeak(double *baskets, int basketCount, int notEqual = -1);

/* Интерполяция параболой */
double parabaloidInterpolation(double* baskets, int basketCount, int maxIndex);

double getGradientValue(double x, double y);

double getGradientDirection(double x, double y);

double getDistance(Descriptor d1, Descriptor d2);
struct lines {
public:
	Descriptor first;
	Descriptor second;
};


// Поиск похожих дескрипторов
vector<lines> findSimilar(vector<Descriptor> d1, vector<Descriptor> d2, double treshhold);
vector<lines> findSimilarNORM(vector<Descriptor> d1, vector<Descriptor> d2, double treshhold);
#endif