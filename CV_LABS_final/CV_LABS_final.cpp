#include "image_functions.h"


void lab4(string filename) {

	Mat original1 = imread(filename);
	Mat original2;
	//original1.convertTo(original2, -1, 2, 0); //increase the contrast by 2
	//original1.convertTo(original2, -1, 4, 0); //increase the contrast by 4
	original1.convertTo(original2, -1, 0.5, 0); //decrease the contrast by 0.5
	//original1.convertTo(original2, -1, 0.25, 0); //decrease the contrast by 0.25

	Mat forcalc1;
	cvtColor(original1, forcalc1, COLOR_BGR2GRAY);
	
	Mat forcalc2;
	cvtColor(original2, forcalc2, COLOR_BGR2GRAY);

	//show_image(forcalc1, "forcalc1");
	//show_image(forcalc2, "forcalc2");
	
	Matrix F1(forcalc1);
	Matrix F2(forcalc2);
	lin_norm(F1, 0, 1);
	lin_norm(F2, 0, 1);

	vector<Pixel> points1 = ANMS(Harris(F1, 5, 0.05, 0), 60);
	vector<Pixel> points2 = ANMS(Harris(F2, 5, 0.05, 0), 60);

	vector<Descriptor> descriptors1 = getDescriptors(forcalc1, points1, 8, 8, 16);

	vector<Descriptor> descriptors2 = getDescriptors(forcalc2, points2, 8, 8, 16);

	vector<lines> matches = findSimilar(descriptors1, descriptors2, 0.80);

	Mat comparsion;
	comparsion.push_back(original1);
	hconcat(comparsion, original2, comparsion);
	for (int i = 0; i < points1.size(); i++)
		circle(comparsion, Point(points1[i].x, points1[i].y), 1, Scalar(0, 0, 255), -1);
	for (int i = 0; i < points2.size(); i++)
		circle(comparsion, Point(points2[i].x + original1.cols, points2[i].y), 1, Scalar(0, 0, 255), -1);

	RNG rng(10);
	for (int i = 0; i < matches.size(); i++)
		line(comparsion, Point(matches[i].first.interPoint.x, matches[i].first.interPoint.y), 
			Point(matches[i].second.interPoint.x + original1.cols, matches[i].second.interPoint.y), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);
	show_image(comparsion, "comparsion");
}

void lab5(string filename) {
	Mat original1 = imread(filename);
	Mat original2 = imread(filename);
	Mat original2_rotated = rotate_img(original2, 180);
	//original2 = rotate_img(original2, 180);
	//show_image(original2, "rotated");
	
	Mat forcalc1;
	cvtColor(original1, forcalc1, COLOR_BGR2GRAY);

	Mat forcalc2; 
	cvtColor(original2_rotated, forcalc2, COLOR_BGR2GRAY);

 
	Matrix F1(forcalc1);
	Matrix F2(forcalc2);
	//Matrix F2_rotated = rotate_mtr(F2, 180);
	//Mat forcalc2_rotated = rotate_img(forcalc2, 180);
	lin_norm(F1, 0, 1);
	lin_norm(F2, 0, 1);


	vector<Pixel> points1 = ANMS(Harris(F1, 5, 0.05, 0), 50);
	vector<Pixel> points2 = ANMS(Harris(F2, 5, 0.05, 5), 50);

	vector<Descriptor> descriptors1 = getDescriptorsInvRot(forcalc1, points1, 8, 8, 16);

	vector<Descriptor> descriptors2 = getDescriptorsInvRot(forcalc2, points2, 8, 8, 16);

	vector<lines> matches = findSimilar(descriptors1, descriptors2, 0.80);

	Mat comparsion;
	comparsion.push_back(original1);
	
	hconcat(comparsion, original2_rotated, comparsion);
	for (int i = 0; i < points1.size(); i++)
		circle(comparsion, Point(points1[i].x, points1[i].y), 1, Scalar(0, 0, 255), -1);
	for (int i = 0; i < points2.size(); i++)
		circle(comparsion, Point(points2[i].x + original1.cols, points2[i].y), 1, Scalar(0, 0, 255), -1);

	RNG rng(10);
	for (int i = 0; i < matches.size(); i++)
		line(comparsion, Point(matches[i].first.interPoint.x, matches[i].first.interPoint.y),
			Point(matches[i].second.interPoint.x + original1.cols, matches[i].second.interPoint.y), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);
	show_image(comparsion, "comparsion");

}
void lab1(Mat original){
	Matrix F(original);
	int Hy_sobel[3][3] = { { -1, 0, 1 },{ -2, 0, 2 },{ -1, 0,1 } };
	int Hx_sobel[3][3] = { { -1, -2, -1 },{ 0, 0, 0 },{ 1,2,1 } };

	lin_norm(F, 0, 1);
	Matrix G_x = derivative(F, Hx_sobel);
	Matrix G_y = derivative(F, Hy_sobel);
	Matrix sobel = sobel_operator(G_x, G_y);
	Matrix gauss = gauss_filter(F, 5, 1);

	lin_norm(G_x, 0, 255);
	lin_norm(G_y, 0, 255);
	lin_norm(sobel, 0, 255);
	lin_norm(gauss, 0, 255);

	show_image(to_image(G_x), "G_x");
	show_image(to_image(G_y), "G_y");
	show_image(to_image(sobel), "sobel");
	show_image(to_image(gauss), "gauss");
}
void lab2(Mat original) {
	Matrix F(original);
	lin_norm(F, 0, 1);
	Matrix* Octaves;
	int s;
	double sig_a, sig_0;

	cout << "input s: ";	cin >> s;
	cout << "input sig_a: "; cin >> sig_a;
	cout << "input sig_0: "; cin >> sig_0;

	Octaves = L(F, s, sig_0, sig_a);
}
void lab3Harris(string filename) {
	cout << "input window: ";			int window;		cin >> window;
	cout << "input number of points: ";	int number;		cin >> number;

	cout << "input harris treshold: ";	double t_har;	cin >> t_har;
	cout << "Harris result:" << endl;
	show_result_harris(filename, window, t_har, number);
	show_result_harris(filename, window, t_har, number, -45);
}
void lab3Moravec(string filename) {
	cout << "input window: ";			int window;		cin >> window;
	cout << "input number of points: ";	int number;		cin >> number;

	cout << "input moravec treshold: ";	double t_mor;	cin >> t_mor;
	cout << "Moravec result:" << endl;
	show_result_moravec(filename, window, t_mor, number);
	show_result_moravec(filename, window, t_mor, number, 10);
}
int main()
{
	string filename = "C:/Users/salom/Desktop/test.jpg";
	Mat original = imread(filename, 0);
	Mat original2 = imread(filename, 0);
	cout << "input lab number: ";		int lab_number; 	cin >> lab_number;
	switch (lab_number)
	{
	case 1: lab1(original); break;
	case 2: lab2(original); break;
	case 3: lab3Harris(filename); break;
	case 4: lab4(filename); break;
	case 5: lab5(filename); break;
	default:
		break;
	}
	

	waitKey();
	return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
