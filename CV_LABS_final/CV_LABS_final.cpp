#include "image_functions.h"


void lab4() {
	string filename = "C:/Users/salom/Desktop/test.jpg";
	Mat original = imread(filename, 0);
	Matrix F(original);
	lin_norm(F, 0, 1);
	Matrix gauss = gauss_filter(F, 5, 1);

	vector<Pixel> points = ANMS(Harris(F, 5, 0.3, 0), 40);
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
	
	cout << "input lab number: ";		int lab_number; 	cin >> lab_number;
	switch (lab_number)
	{
	case 1: lab1(original); break;
	case 2: lab2(original); break;
	case 3: lab3Harris(filename);
		lab3Moravec(filename); break;
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
