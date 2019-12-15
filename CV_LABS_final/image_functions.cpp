#include "image_functions.h"
#define PI 3.14159265
int Hx_sobel[3][3] = { { 1, 0, -1 },
{ 2, 0, -2 },
{ 1, 0,-1 } };
int Hy_sobel[3][3] = { { 1, 2, 1 },
{ 0, 0, 0 },
{ -1,-2,-1 } };

Matrix::Matrix() {
	height = 0;
	width = 0;
	values = NULL;
}
Matrix::Matrix(int H, int W) {
	height = H;
	width = W;
	values = new double* [height];
	for (int i = 0; i < H; i++) {
		values[i] = new double[width];
	}
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			values[y][x] = 0.0;
		}
	}
}
Matrix::Matrix(Mat image) {
	height = image.rows;
	width = image.cols;
	values = new double* [height];
	for (int y = 0; y < height; y++)
		values[y] = new double[width];

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			values[y][x] = (double)image.at<uchar>(y, x);
		}
	}
}
void Matrix::to_zero() {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			values[y][x] = 0.0;
		}
	}
}
Matrix::Matrix(const Matrix& obj) {
	height = obj.height;
	width = obj.width;
	values = new double* [height];
	for (int y = 0; y < height; y++)
		values[y] = new double[width];

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			values[y][x] = obj.values[y][x];
		}
	}
}
Matrix& Matrix::operator=(const Matrix& obj) {
	if (this != &obj) {

		height = obj.height;
		width = obj.width;
		values = new double* [height];
		for (int y = 0; y < height; y++)
			values[y] = new double[width];

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				values[y][x] = obj.values[y][x];
			}
		}
	}
	return *this;
}
Matrix operator *(Matrix A, Matrix B) {
	Matrix Result(A.height, B.width);

	for (int y = 0; y < A.height; y++) {
		for (int x = 0; x < B.width; x++) {
			double sum = 0;
			for (int k = 0; k < B.height; k++) {
				sum += A.values[y][k] * B.values[k][x];
			}
			Result.values[y][x] = sum;
		}
	}
	return Result;
}

Matrix::~Matrix() {
	for (int y = 0; y < height; y++) {
		delete[] values[y];
	}
	delete[] values;
}

void lin_norm(Matrix& F, int newMin, int newMax) {
	double min = F.values[0][0];
	double max = F.values[0][0];

	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			if (F.values[y][x] > max) {
				max = F.values[y][x];
			}

			if (F.values[y][x] < min) {
				min = F.values[y][x];
			}
		}
	}

	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			F.values[y][x] = (F.values[y][x] - min) * (newMax - newMin) / (max - min) + newMin;
		}
	}
}
Mat to_image(Matrix F) {
	Mat img(F.height, F.width, CV_8UC1, Scalar(0));
	for (int y = 0; y < F.height; y++)
		for (int x = 0; x < F.width; x++)
			img.at<uchar>(y, x) = (int)F.values[y][x];
	return img;
}
void show_image(Mat F, string name) {
	namedWindow(name);
	imshow(name, F);
}
void write_image(Mat img, string name) {
	imwrite(name + ".jpg", img);
}

int check_edge(int x, int x_i, int max) {
	if (x + x_i < 0) return x - x_i;
	else if (x + x_i > max) return max - x_i;
	else return x;
}

int repeat_edge(int x, int x_i, int max) {
	if (x + x_i < 0) return 0;
	else if (x + x_i > max-1) return max-1;
	else return x + x_i;
}

Matrix derivative(Matrix F, int(&H)[3][3]) {
	Matrix G(F.height, F.width);

	int K = 0;

	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			double sum = 0;
			for (int y_i = -1; y_i <= 1; y_i++) {
				for (int x_i = -1; x_i <= 1; x_i++)
				{
					int j = check_edge(y, y_i, F.height - 1);
					int i = check_edge(x, x_i, F.width - 1);

					sum += H[1 + x_i][1 + y_i] * F.values[j + y_i][i + x_i];
				}
			}
			G.values[y][x] = sum;
		}
	}

	return G;
}

Matrix sobel_operator(Matrix G_x, Matrix G_y) {
	Matrix sobel(G_x.height, G_x.width);

	for (int y = 0; y < G_x.height; y++) {
		for (int x = 0; x < G_x.width; x++) {
			sobel.values[y][x] = sqrt(pow(G_x.values[y][x], 2) + pow(G_y.values[y][x], 2));
		}
	}
	return sobel;
}

Matrix gauss_filter(Matrix F, int ker_size, double sigma) {
	if (sigma == 0) return F;
	Matrix gauss(F.height, F.width);

	Matrix kernel = gauss_weight(ker_size, sigma);
	int R = ker_size / 2;
	double s = 2 * sigma * sigma;
	double sum = 0;

	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			for (int y_i = -R; y_i <= R; y_i++) {
				for (int x_i = -R; x_i <= R; x_i++)
				{
					int j = check_edge(y, y_i, F.height - 1);
					int i = check_edge(x, x_i, F.width - 1);
					sum += kernel.values[R + x_i][R + y_i] * F.values[j + y_i][i + x_i];
				}
			}
			gauss.values[y][x] = sum;
			sum = 0;
		}

	}
	return gauss;
}

Matrix gauss_weight(int ker_size, double sigma) {
	Matrix kernel(ker_size, ker_size);

	int R = ker_size / 2;
	double s = 2 * sigma * sigma;

	for (int x = -R; x <= R; x++) {
		for (int y = -R; y <= R; y++) {
			kernel.values[x + R][y + R] = (1 / (3.14 * s)) * exp(-(x * x + y * y) / s);
		}
	}
	lin_norm(kernel, 0, 1);
	return kernel;
}

Matrix downsample(Matrix F) {
	Matrix down(F.height / 2, F.width / 2);
	for (int y = 0; y < F.height / 2; y++) {
		for (int x = 0; x < F.width / 2; x++) {
			down.values[y][x] = F.values[y * 2][x * 2];
		}
	}
	return down;
}

int number_of_layers(Matrix F) {
	int H = F.height, W = F.width;
	int layers_numb = 0;
	while (H > 100 && W > 100) {
		H /= 2;
		W /= 2;
		layers_numb++;
	}
	return layers_numb;
}

Matrix* L(Matrix F, int s, double sig_0, double sig_a) {

	int L_n = number_of_layers(F);

	double k = pow(2, 1.0 / s);
	double* sig = new double[s + 1];
	sig[0] = sig_0;
	for (int i = 1; i <= s; i++) {
		sig[i] = sig[0] * pow(k, i);
	}

	Matrix* Octaves = new Matrix[L_n * s + L_n];

	double g = sqrt(abs(sig_0 * sig_0 - sig_a * sig_a));
	Octaves[0] = gauss_filter(F, 5, g);

	for (int i = 0; i < L_n * s + L_n; i += s + 1) {
		for (int j = 1; j <= s; j++) {
			double g_sig = sqrt(abs(sig[j] * sig[j] - sig[j - 1] * sig[j - 1]));
			Octaves[i + j] = gauss_filter(Octaves[i + j - 1], 5, g_sig);
		}
		Octaves[i + s + 1] = downsample(Octaves[i + s]);
	}
	//write to files
	int oct = -1;
	for (int i = 0; i < L_n * s + L_n; i++) {
		if (i % (s + 1) == 0) oct++;
		string str;
		str = to_string(oct) + "oct_" + to_string(i % (s + 1)) + "level";
		lin_norm(Octaves[i], 0, 255);
		write_image(to_image(Octaves[i]), str);
	}

	return Octaves;
}

vector <Pixel> Moravec(Matrix F, int sizeW, double treshold, double angle) {

	Matrix resp(F.height, F.width);
	int R = sizeW / 2;
	double* V = new double[8];
	int uv[16] = { -1,-1,-1,0,-1,1,0,-1,0,1,1,-1,1,0,1,1 };

	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			double sum = 0;
			for (int i = 0; i < 15; i += 2) {
				for (int y_i = -R; y_i <= R; y_i++) {
					for (int x_i = -R; x_i <= R; x_i++) {

						int a1 = check_edge(y, y_i + uv[i], F.height - 1);
						int b1 = check_edge(x, x_i + uv[i + 1], F.width - 1);
						int a2 = check_edge(y, y_i, F.height - 1);
						int b2 = check_edge(x, x_i, F.width - 1);

						double ls = F.values[a1 + y_i + uv[i]][b1 + x_i + uv[i + 1]];
						double l = F.values[a2 + y_i][b2 + x_i];
						sum += pow(ls - l, 2);
					}
				}
				V[i / 2] = sum;
				sum = 0;
			}
			double min = V[0];
			for (int i = 0; i < 8; i++)
			{
				if (V[i] <= min)
				{
					min = V[i];
				}
			}
			resp.values[y][x] = min;
		}
	}
	lin_norm(resp, 0, 1);
	Matrix to_show_resp = resp;
	lin_norm(to_show_resp, 0, 255);
	show_image(to_image(to_show_resp), "map_moravec " + to_string((int)angle));

	vector<Pixel> result;

	for (int y = 0; y < resp.height; y++) {
		for (int x = 0; x < resp.width; x++) {
			if (resp.values[y][x] > treshold&& find_mx(resp, sizeW, x, y)) {

				Pixel temp;
				temp.y = y;
				temp.x = x;
				temp.value = resp.values[y][x];
				temp.interest = true;

				result.push_back(temp);
			}
		}
	}
	return result;
}

vector<Pixel> Harris(Matrix F, int sizeW, double treshold, double angle) {
	Matrix Ix = derivative(F, Hx_sobel);
	Matrix Iy = derivative(F, Hy_sobel);
	Matrix gauss_w = gauss_weight(sizeW, 1);

	Matrix resp(F.height, F.width);
	Matrix E(F.height, F.width);
	int R = sizeW / 2;
	double k = 0.05;

	double A, B, C;
	for (int y = 0; y < F.height; y++) {
		for (int x = 0; x < F.width; x++) {
			if (x == 47 && y == 42)
				int stop = 1;
			A = B = C = 0;
			for (int y_i = -R; y_i <= R; y_i++) {
				for (int x_i = -R; x_i <= R; x_i++) {

					int a = check_edge(y, y_i, F.height - 1);
					int b = check_edge(x, x_i, F.width - 1);

					A += gauss_w.values[y_i + R][x_i + R] * Ix.values[y_i + a][x_i + b] * Ix.values[y_i + a][x_i + b];
					B += gauss_w.values[y_i + R][x_i + R] * Ix.values[y_i + a][x_i + b] * Iy.values[y_i + a][x_i + b];
					C += gauss_w.values[y_i + R][x_i + R] * Iy.values[y_i + a][x_i + b] * Iy.values[y_i + a][x_i + b];
				}
			}
			double det = A * C - pow(B, 2);
			double trace = A + C;
			double response = det - k * pow(trace, 2);
			if (response < 0) resp.values[y][x] = 0;
			else resp.values[y][x] = response;
		}
	}

	lin_norm(resp, 0, 1);
	Matrix to_show_resp = resp;
	lin_norm(to_show_resp, 0, 255);
	show_image(to_image(to_show_resp), "map_harris " + to_string((int)angle));

	vector<Pixel> result;

	for (int y = 0; y < resp.height; y++) {
		for (int x = 0; x < resp.width; x++) {
			if (resp.values[y][x] > treshold&& find_mx(resp, sizeW, x, y)) {
				Pixel temp;
				temp.y = y;
				temp.x = x;
				temp.value = resp.values[y][x];
				temp.interest = true;

				result.push_back(temp);
			}
		}
	}
	return result;
}

double Matrix_trace(Matrix F) {
	if (F.height != F.width) {
		cout << "Not square Matrix!";
		return 0;
	}
	else {
		double sum = 0;
		for (int i = 0; i < F.height; i++)
			sum += F.values[i][i];
		return sum;
	}
}

double Matrix_det(Matrix F) {
	if (F.height != F.width) {
		cout << "Not square Matrix!";
		return 0;
	}
	else {
		double sum = 0;
		for (int i = 0; i < F.height; i++)
			sum += F.values[i][i];
		return sum;
	}
}

vector<Pixel> ANMS(vector<Pixel> result, int number) {
	int r = 0;
	int count = result.size();
	while (count >= number) {
		r++;
		for (int i = 0; i < result.size(); i++) {
			for (int j = i + 1; j < result.size(); j++) {
				if (i != j && result[i].interest && result[j].interest) {
					double P = sqrt(pow(result[i].x - result[j].x, 2) + pow(result[i].y - result[j].y, 2));
					if ((P < r) && (result[i].value < 0.9 * result[j].value)) {
						result[i].interest = false;
						count--;
					}
				}
			}
		}
	}
	vector<Pixel> result_final;
	for (int j = 0; j < result.size(); j++)
		if (result[j].interest)
			result_final.push_back(result[j]);
	cout << "\tRadius=" << r << "\tPoints=" << result.size() << "\tBest Points=" << result_final.size() << endl;

	return result_final;
}

vector<Pixel> ANMS_rad(vector<Pixel> result, int rad) {
	for (int i = 0; i < result.size(); i++) {
		for (int j = i + 1; j < result.size(); j++) {
			if (i != j && result[i].interest && result[j].interest) {
				double P = sqrt(pow(result[i].x - result[j].x, 2) + pow(result[i].y - result[j].y, 2));
				if ((P < rad) /*&& (result[i].value < 0.9*result[j].value)*/) {
					result[i].interest = false;
				}
			}
		}
	}

	vector<Pixel> result_final;
	for (int j = 0; j < result.size(); j++)
		if (result[j].interest)
			result_final.push_back(result[j]);
	cout << "\tRadius=" << rad << "\tPoints=" << result.size() << "\tBest Points=" << result_final.size() << endl;

	return result_final;
}


Matrix rotate_mtr(Matrix F, double angle) {

	int x_center = F.height / 2, y_center = F.width / 2;
	double aRad = angle * PI / 180;
	Matrix rot(F.height, F.width);
	int ty, tx;

	for (int y = 0; y < rot.height; y++) {
		for (int x = 0; x < rot.width; x++) {

			tx = cos(aRad) * (x - x_center) - sin(aRad) * (y - y_center) + x_center;
			ty = sin(aRad) * (x - x_center) + cos(aRad) * (y - y_center) + y_center;

			if (tx >= 0 && tx < rot.width && ty >= 0 && ty < rot.height) {

				rot.values[y][x] = F.values[ty][tx];
			}
		}
	}
	return rot;
}
Mat rotate_img(Mat F, double angle) {
	// матрицы трансформации
	Mat rot_mat(2, 3, CV_8UC1);
	Mat rot = F;
	// вращение относительно центра изображения
	Point2d center = Point2d(F.cols / 2, F.rows / 2);
	double scale = 1;
	rot_mat = getRotationMatrix2D(center, angle, scale);

	warpAffine(F, rot, rot_mat, Size(F.cols, F.rows));

	return rot;
}

void show_result_moravec(string filename, int window, double t_mor, int number, double angle) {
	Mat Mor = imread(filename);
	Mat Mor_ANMS = imread(filename);

	Matrix F(imread(filename, 0));
	lin_norm(F, 0, 1);
	if (angle != 0) {
		F = rotate_mtr(F, angle);
		Mor = rotate_img(Mor, angle);
		Mor_ANMS = rotate_img(Mor_ANMS, angle);
	}
	Matrix sobel = sobel_operator(derivative(F, Hx_sobel), derivative(F, Hy_sobel));
	vector <Pixel> moravec_tr = Moravec(sobel, window, t_mor, angle);

	for (int i = 0; i < moravec_tr.size(); i++)
		circle(Mor, Point(moravec_tr[i].x, moravec_tr[i].y), 1, Scalar(0, 0, 255), -1);
	show_image(Mor, "p_Moravec " + to_string((int)angle));

	vector <Pixel> mor_anms = ANMS(moravec_tr, number);
	for (int i = 0; i < mor_anms.size(); i++)
		circle(Mor_ANMS, Point(mor_anms[i].x, mor_anms[i].y), 1, Scalar(0, 0, 255), -1);
	show_image(Mor_ANMS, "p_Moravec_ANMS " + to_string((int)angle));
}

void show_result_harris(string filename, int window, double t_har, int number, double angle) {
	Mat Har = imread(filename);
	Mat Har_ANMS = imread(filename);

	Matrix F(imread(filename, 0));
	lin_norm(F, 0, 1);
	if (angle != 0) {
		F = rotate_mtr(F, angle);
		Har = rotate_img(Har, angle);
		Har_ANMS = rotate_img(Har_ANMS, angle);
	}
	vector <Pixel> harris_tr = Harris(F, window, t_har, angle);
	for (int i = 0; i < harris_tr.size(); i++) {
		circle(Har, Point(harris_tr[i].x, harris_tr[i].y), 1, Scalar(0, 0, 255), -1);
	}
	show_image(Har, "p_Harris " + to_string((int)angle));

	vector <Pixel> harris_anms = ANMS(harris_tr, number);
	for (int i = 0; i < harris_anms.size(); i++)
		circle(Har_ANMS, Point(harris_anms[i].x, harris_anms[i].y), 1, Scalar(0, 0, 255), -1);
	show_image(Har_ANMS, "p_Harris_ANMS " + to_string((int)angle));

	//vector <Pixel> harris_rad = ANMS_rad(harris_tr, 5);
	//Mat Har_Rad = imread(filename);
	/*for (int i = 0; i<harris_anms_rad.size(); i++)
	circle(H3, Point(harris_anms_rad[i].x, harris_anms_rad[i].y), 1, Scalar(0, 0, 255), -1);
	show_image(Har_Rad, "p_harris_ANMS_rad5");*/
}

bool find_mx(Matrix F, int sizeW, int x, int y) {
	int R = sizeW / 2;
	double mx = F.values[y][x];
	for (int i = -R; i <= R; i++) {
		for (int j = -R; j <= R; j++) {
			int q = check_edge(x, i, F.width - 1);
			int w = check_edge(y, j, F.height - 1);
			if (F.values[w + j][q + i] > mx) mx = F.values[w + j][q + i];
		}
	}
	if (F.values[y][x] < mx)
		return false;
	else return true;
}
double Clamp(double min, double max, double value)
{
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
};

void Descriptor::normalize() {
	double length = 0;
	for (int i = 0; i < N; i++)
		length += data[i] * data[i];

		length = sqrt(length);

		for (int i = 0; i < N; i++)
			data[i] /= length;
}
void Descriptor::clampData(double min, double max)
{
	for (int i = 0; i < N; i++)
		data[i] = Clamp(min, max, data[i]);
}

vector<Descriptor> get_Descriptors(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int histCountInLine)
{
	double sector = 2 * PI / basketCount; //размер одной корзины в гистограмме
	double halfSector = PI / basketCount; // размер половины одной корзины в гистограмме
	int histStep = radius * 2 / histCountInLine; //шаг гистограммы
	int histCount = histCountInLine * histCountInLine; //общее количество гистограмм в дескрипторе


	Matrix image_dx = derivative(image, Hx_sobel);
	Matrix image_dy = derivative(image, Hy_sobel);

	vector<Descriptor> descriptors;
	for (int k = 0; k < interestPoints.size(); k++)
	{
		descriptors.push_back(Descriptor(histCount * basketCount, interestPoints[k]));

		for (int i = - radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				int coordY = repeat_edge(interestPoints[k].y, i, image_dx.height);
				int coordX = repeat_edge(interestPoints[k].x, j, image_dx.width);
				// get Gradient
				double gradient_X = image_dx.values[coordY][coordX];
				double gradient_Y = image_dy.values[coordY][coordX];

				// get value and phi
				double value = getGradientValue(gradient_X, gradient_Y);
				double phi = getGradientDirection(gradient_X, gradient_Y);

		
				int leftBasketIndex, rightBasketIndex;

				double leftBasketKoef;
				double rightBasketKoef;
				// получаем индекс корзины в которую входит phi и смежную с ней
				if (phi < (sector * 0) + halfSector) {
					leftBasketIndex = basketCount - 1;
					rightBasketIndex = 0;
					// (sector * 0) + halfSector - середина первой корзины
					rightBasketKoef = (((sector * 0) + halfSector) - phi) / sector;
					leftBasketKoef = 1 - rightBasketKoef;
				}
				else if (phi > (sector * (basketCount - 1) + halfSector)) {
					leftBasketIndex = basketCount - 1;
					rightBasketIndex = 0;

					// (sector * (basketCount-1)) + halfSector - середина последней корзины
					leftBasketKoef = (phi - (sector * leftBasketIndex + halfSector)) / sector;
					rightBasketKoef = 1 - leftBasketKoef;
				}
				else {
					leftBasketIndex = (int)((phi - halfSector) / sector);
					rightBasketIndex = leftBasketIndex + 1;

					leftBasketKoef = (phi - (sector * leftBasketIndex + halfSector)) / sector;
					rightBasketKoef = 1 - leftBasketKoef;
				}


				// распределяем L(value)
				double mainBasketValue = (1 - leftBasketKoef) * value;
				double sideBasketValue = rightBasketKoef * value;

				// вычисляем индекс куда записывать значения
				int tmp_i = (i + radius) / histStep;
				int tmp_j = (j + radius) / histStep;

				int indexMain = (tmp_i * histCountInLine + tmp_j) * basketCount + leftBasketIndex;
				int indexSide = (tmp_i * histCountInLine + tmp_j) * basketCount + rightBasketIndex;

				if (indexMain >= descriptors[k].N)
					indexMain = 0;

				if (indexSide >= descriptors[k].N)
					indexSide = 0;

				// записываем значения
				descriptors[k].data[indexMain] += mainBasketValue;
				descriptors[k].data[indexSide] += sideBasketValue;
			}
		}
		descriptors[k].normalize();
		descriptors[k].clampData(0, 0.2);
		descriptors[k].normalize();
	}
	return descriptors;
}

vector<Descriptor> GET_NEW_DescriptorsInvRot(Mat image, vector<Pixel> interestPoints, int radius, int basketCount, int histCountInLine)
{
	//radius - окрестность интересной точки
	double sector = 2 * PI / basketCount; //размер одной корзины в гистограмме
	double halfSector = PI / basketCount; // размер половины одной корзины в гистограмме
	int histStep = radius*2 / histCountInLine; //шаг гистограммы
	int histCount = histCountInLine * histCountInLine; //общее количество гистограмм в дескрипторе

	Matrix image_dx = derivative(image, Hx_sobel);
	Matrix image_dy = derivative(image, Hy_sobel);

	vector<Descriptor> descriptors;
	Matrix Gauss = gauss_weight(2 * radius + 1, (double)radius / 3);

	for (int k = 0; k < interestPoints.size(); k++)
	{
		descriptors.push_back(Descriptor(histCount * basketCount, interestPoints[k]));
		vector<double> peaks = GET_NEW_PointOrientation(image_dx, image_dy, interestPoints[k], radius);    // Ориентация точки

		for (int p = 0; p < peaks.size(); p++) {
			double phiRotate = peaks[p];

			for (int y_i = -radius; y_i <= radius; y_i++)
			{
				for (int x_i = -radius; x_i <= radius; x_i++)
				{

					// координаты
					int coord_X = repeat_edge(interestPoints[k].x, x_i, image_dx.width);
					int coord_Y = repeat_edge(interestPoints[k].y, y_i, image_dx.height);

					// градиент
					double gradient_X = image_dx.values[coord_Y][coord_X];
					double gradient_Y = image_dy.values[coord_Y][coord_X];

					// получаем значение(домноженное на Гаусса) и угол
					double value = getGradientValue(gradient_X, gradient_Y) * Gauss.values[y_i + radius][x_i + radius];
					double phi = getGradientDirection(gradient_X, gradient_Y);

					// Определяем гистограмму куда мы будем записывать
					int x_i_Rotate = (int)round((x_i)*cos(phiRotate) + (y_i)*sin(phiRotate));
					int y_i_Rotate = (int)round(-(x_i)*sin(phiRotate) + (y_i)*cos(phiRotate));
					// отбрасываем
					if (x_i_Rotate < -radius || y_i_Rotate < -radius || x_i_Rotate >= radius || y_i_Rotate >= radius)
					{
						continue;
					}
					int tmp_i = (x_i_Rotate + radius) / histStep;
					int tmp_j = (y_i_Rotate + radius) / histStep;

					// поиск итогового угла
					double finalPhi = phi - phiRotate;
					if (finalPhi < 0)
						finalPhi = 2 * PI + finalPhi;
					if (finalPhi > 2 * PI)
						finalPhi = finalPhi - 2*PI;

					int leftBasketIndex;
					int rightBasketIndex;

					// Коэффициент принадлежности к корзине (0,1)
					double leftBasketKoef;
					double rightBasketKoef;
					if (finalPhi < (sector * 0) + halfSector) {
						leftBasketIndex = basketCount - 1;
						rightBasketIndex = 0;
						// (sector * 0) + halfSector - середина первой корзины
						rightBasketKoef = (((sector * 0) + halfSector) - finalPhi) / sector;
						leftBasketKoef = 1 - rightBasketKoef;
					}
					else if (finalPhi > (sector * (basketCount - 1) + halfSector)) {
						leftBasketIndex = basketCount - 1;
						rightBasketIndex = 0;

						// (sector * (basketCount-1)) + halfSector - середина последней корзины
						leftBasketKoef = (finalPhi - (sector * leftBasketIndex + halfSector)) / sector;
						rightBasketKoef = 1 - leftBasketKoef;
					}
					else {
						leftBasketIndex = (int)((finalPhi - halfSector) / sector);
						rightBasketIndex = leftBasketIndex + 1;

						leftBasketKoef = (finalPhi - (sector * leftBasketIndex + halfSector)) / sector;
						rightBasketKoef = 1 - leftBasketKoef;
					}


					// распределяем L(value)
					double mainBasketValue = (1 - leftBasketKoef) * value;
					double sideBasketValue = rightBasketKoef * value;


					//////////////////
					// вычисляем индекс куда записывать значения
					int indexMain = (tmp_i * histCountInLine + tmp_j) * basketCount + leftBasketIndex;
					int indexSide = (tmp_i * histCountInLine + tmp_j) * basketCount + rightBasketIndex;

					// записываем значения
					descriptors[k].data[indexMain] += mainBasketValue;
					descriptors[k].data[indexSide] += sideBasketValue;
				}
			}
			descriptors[k].normalize();
			descriptors[k].clampData(0, 0.2);
			descriptors[k].normalize();
		}
	}
	return descriptors;
}

vector<double> GET_NEW_PointOrientation(Matrix image_dx, Matrix image_dy, Pixel point, int radius, int basketCount)
{

	double sector = 2 * PI / basketCount;
	double halfSector = sector / 2;
	
	int leftBasketIndex;
	int rightBasketIndex;

	double* baskets = new double[basketCount];

	for (int i = 0; i < basketCount; i++)
		baskets[i] = 0;

	Matrix Gauss = gauss_weight(2 * radius + 1, (double)radius / 3);

	for (int y_i = -radius; y_i <= radius; y_i++)
	{
		for (int x_i = -radius; x_i <= radius; x_i++)
		{

			int coord_X = repeat_edge(point.x, x_i, image_dx.width - 1);
			int coord_Y = repeat_edge(point.y, y_i, image_dx.height -1);

			// градиент

			double gradient_X = image_dx.values[coord_Y][coord_X];
			double gradient_Y = image_dy.values[coord_Y][coord_X];

			// получаем значение(домноженное на Гаусса) и угол
			double value = getGradientValue(gradient_X, gradient_Y) * Gauss.values[y_i + radius][x_i + radius];
			double phi = getGradientDirection(gradient_X, gradient_Y);

			// Коэффициент принадлежности к корзине (0,1)
			double leftBasketKoef;
			double rightBasketKoef;
			if (phi < (sector * 0) + halfSector) {
				leftBasketIndex = basketCount - 1;
				rightBasketIndex = 0;
				// (sector * 0) + halfSector - середина первой корзины
				rightBasketKoef = (((sector * 0) + halfSector) - phi) / sector;
				leftBasketKoef = 1 - rightBasketKoef;
			}
			else if (phi > (sector * (basketCount - 1) + halfSector)) {
				leftBasketIndex = basketCount - 1;
				rightBasketIndex = 0;
				
				// (sector * (basketCount-1)) + halfSector - середина последней корзины
				leftBasketKoef = (phi - (sector * leftBasketIndex + halfSector)) / sector;
				rightBasketKoef = 1 - leftBasketKoef;
			}
			else {
				leftBasketIndex = (int)((phi - halfSector) / sector);
				rightBasketIndex = leftBasketIndex + 1;

				leftBasketKoef = (phi - (sector * leftBasketIndex + halfSector)) / sector;
				rightBasketKoef = 1 - leftBasketKoef;
			}

			baskets[leftBasketIndex] += rightBasketKoef * value;
			baskets[rightBasketIndex] += leftBasketKoef * value;
		}
	}

	// Ищем Пики
	double peak_1, peak_2;
	int peak_1_index = 0, peak_2_index = 0;
	peak_1 = baskets[0];
	peak_2 = baskets[0];
	for (int i = 1; i < basketCount; i++) {
		if (baskets[i] > peak_1) {
			peak_2 = peak_1;
			peak_2_index = peak_1_index;
			peak_1 = baskets[i];
			peak_1_index = i;
		}
	}
	//хотя бы peak_1 должен быть!
	vector<double> peaks;
	if (peak_2 > peak_1 * 0.8 && peak_1_index != peak_2_index)
	{ // Если второй пик не ниже 80%
		peaks.push_back(peak_1_index * sector + halfSector);
		peaks.push_back(peak_2_index * sector + halfSector);
	}
	else
	{
		peaks.push_back(peak_1_index * sector + halfSector);
	}
	return peaks;
}


double getGradientValue(double x, double y)
{
	return sqrt(x * x + y * y);
}

double getGradientDirection(double x, double y) {
	double angle = atan2(x, y);
	return angle > 0 ? angle : PI*2 + angle;
	//return atan2(x, y) + PI; 
}
vector<lines> findSimilarNORM(vector<Descriptor> d1, vector<Descriptor> d2, double treshhold)
{
	vector<lines> similar;
	for (int i = 0; i < d1.size(); i++)
	{

		int match1_index = 0;
		double match1_value = getDistance(d1[i], d2[0]);
		int match2_index = 1;
		double match2_value = getDistance(d1[i], d2[1]);

		if (match2_value < match1_value) {
			int temp_index = match1_index;
			double tempvalue = match1_value;
			match1_index = match2_index;
			match1_value = match2_value;
			match2_index = temp_index;
			match2_value = tempvalue;
		}

		for (int j = 2; j < d2.size(); j++) {
			double temp = getDistance(d1[i], d2[j]);
			if (temp < match1_value) {
				match2_index = match1_index;
				match2_value = match1_value;
				match1_index = j;
				match1_value = temp;
			}
		}

		if (match1_value / match2_value > treshhold)
		{
			continue;      // отбрасываем
		}
		else
		{
			lines l;
			l.first = d1[i];
			l.second = d2[match1_index];
			similar.push_back(l);
		}
	}
	return similar;
}

double getDistance(Descriptor d1, Descriptor d2)
{
	double result = 0;
	for (int i = 0; i < d1.N; i++)
	{
		double tmp = d1.data[i] - d2.data[i];
		result += tmp * tmp;
	}
	return sqrt(result);
}

//Matrix *L(Matrix *F, int s, double sig_0, double sig_a) {
//
//	Matrix *Octaves = new Matrix[10];
//
//	double g = sqrt(abs(sig_0 * sig_0 - sig_a * sig_a));
//	Octaves[0] = gauss_filter(F, 5, g);
//
//	for (int i = 0; i < L_n*s + L_n; i += s + 1) {
//		for (int j = 1; j <= s; j++) {
//			double g_sig = sqrt(abs(sig[j] * sig[j] - sig[j - 1] * sig[j - 1]));
//			Octaves[i + j] = gauss_filter(Octaves[i + j - 1], 5, g_sig);
//		}
//		Octaves[i + s + 1] = downsample(Octaves[i + s]);
//	}
//	//write to files
//	int oct = -1;
//	for (int i = 0; i < L_n * s + L_n; i++) {
//		if (i % (s + 1) == 0) oct++;
//		string str;
//		str = to_string(oct) + "oct_" + to_string(i % (s + 1)) + "level";
//		lin_norm(Octaves[i], 0, 255);
//		write_image(to_image(Octaves[i]), str);
//	}
//
//	return Octaves;
//}