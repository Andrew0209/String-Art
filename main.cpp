#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;
using namespace cv;
#define PI acos(-1)
#define uchar unsigned char 

struct Pixel {
    int x;
    int y;
    uchar color;
    Pixel(int xPos = 0, int yPos = 0, uchar pixelColor = 0) {
        this->x = xPos;
        this->y = yPos;
        this->color = pixelColor;
    }
};

void viewMatrix(vector <vector <uchar>>& image, Mat& output);
void linePosition(Pixel Pixel1, Pixel Pixel2, vector <Pixel>& line);
void updateImage(vector<vector<uchar>>& Image, const vector <Pixel>& changes);
vector<Pixel> createCircle(int x, int y, int R, uchar color = 255);
double difference(const vector<Pixel>& line, const vector<vector<uchar>>& OriginalImage, const vector<vector<uchar>>& NewImage, 
    const vector<vector<double>>& Scale = {});
int dist(int a, int b, int MaxNum) { return min(abs(a - b), MaxNum - abs(a - b)); }
void stepDistribution(vector<vector<double>>& scaleMatrix, double step = 0.5, int R = -1, bool Normalize = false);
void normalize(vector<vector<double>>& scaleMatrix);
void onMouse(int event, int x, int y, int flags, void* userdata);

int main(){
    const Mat OriginalImg = imread("C:/Users/37602/source/graphic/test-images/test8.jpg");
    Mat resizedImage;
    resize(OriginalImg, resizedImage, Size(1000, 1000));
    Mat GrayImage;
    cvtColor(resizedImage, GrayImage, COLOR_BGR2GRAY);
    imshow("Gray image", GrayImage);
    waitKey(1);
    const int ImageWidth = GrayImage.cols;
    const int ImageHeight = GrayImage.rows;
    cout << "Image Size: " << ImageWidth << ' ' << ImageHeight << '\n';

    // Image matrix
    vector <vector <uchar>> image(ImageHeight, vector <uchar> (ImageWidth));
    for (int i = 0; i < ImageHeight; i++) {
        for (int j = 0; j < ImageWidth; j++) {
            int pix = int(GrayImage.at<uchar>(i, j));
            if (pix != 255)pix = max(int(pix * 0.9 - 20), 0);
            image[i][j] = pix;
        }
    }

    // String art matrix
    vector <vector <uchar>> stringArt(ImageHeight, vector <uchar>(ImageWidth));
    for (int i = 0; i < ImageHeight; i++) {
        for (int j = 0; j < ImageWidth; j++) {
            stringArt[i][j] = 255;
        }
    }

    // setup string art
    const int Radius = min(ImageHeight, ImageWidth) / 2 - 10; //  - some small delta
    cout << "R = " << Radius << '\n';

    int NailsNumber = 200;
    vector <Pixel> Nails(NailsNumber);
    double Angle = 2 * PI / NailsNumber;
    for (int i = 0; i < NailsNumber; i++) {
        int x = Radius * cos(Angle * i) + ImageWidth / 2;
        int y = Radius * sin(Angle * i) + ImageHeight / 2;
        Nails[i] = Pixel(x, y, 255);
        //cout << cos(Angle * i) << ' ' << sin(Angle * i) << '\n';
    }
    vector<Pixel>circle;
    for (int i = 0; i < NailsNumber; i++) {
        circle = createCircle(Nails[i].x, Nails[i].y, 2,Nails[i].color);
        updateImage(stringArt, circle);
    }

    // Scale matrix
    vector <vector <double>> scaleMatrix(ImageHeight, vector <double>(ImageWidth, 1.0));

    stepDistribution(scaleMatrix);
    Mat DistributionImage;
    cvtColor(GrayImage, DistributionImage, COLOR_GRAY2RGB);
    namedWindow("Distribution");
    setMouseCallback("Distribution", onMouse, &DistributionImage);
    while (true) {
    imshow("Distribution", DistributionImage);
    if (waitKey(1) == 27) break;  // Exit on ESC key press
    }
    destroyWindow("Distribution");
    for (int i = 0; i < ImageHeight; i++)
        for (int j = 0; j < ImageWidth; j++)
            if (DistributionImage.at<Vec3b>(i, j) == Vec3b(0, 0, 200))
                scaleMatrix[i][j] += 2;

    double length = 0;
    // Main algorythm - greedy algorythm
    int startNailIndex = 0, endNailIndex = 0;
    int IterationNumber = 4000;
    vector <bool> deadEnds(NailsNumber, false);
    int deadEndNails = 0;
    vector<int> nailsOrder(IterationNumber, -1);
    //for (int iteration = 0; iteration < IterationNumber; iteration++) {
    int iteration = 0;
    while(iteration < IterationNumber){
        vector<Pixel>line;
        vector<Pixel>optimalLine;
        double optimalDiff = 1;
        for (int i = 0; i < NailsNumber; i++) {
            if (dist(startNailIndex, i, NailsNumber) > 10) {
                linePosition(Nails[startNailIndex], Nails[i], line);
                double diff = difference(line, image, stringArt, scaleMatrix);
                //cout << diff << '\n';
                if (diff > optimalDiff) {
                    optimalDiff = diff;
                    optimalLine = line;
                    endNailIndex = i;
                }
            }
        }
        if (startNailIndex != endNailIndex) {
            // update line length
            double dx = (Nails[startNailIndex].x - Nails[endNailIndex].x);
            double dy = (Nails[startNailIndex].y - Nails[endNailIndex].y);
            length += sqrt(dx * dx + dy * dy) / Radius;
            startNailIndex = endNailIndex;
        }
        else {
            if (!deadEnds[startNailIndex]) {
                deadEndNails++;
                deadEnds[startNailIndex] = true; 
            }
            linePosition(Nails[startNailIndex], Nails[(startNailIndex + 1) % NailsNumber], optimalLine);
            startNailIndex = (startNailIndex + 1) % NailsNumber;
            // update line length
            double dx = (Nails[startNailIndex].x - Nails[endNailIndex].x);
            double dy = (Nails[startNailIndex].y - Nails[endNailIndex].y);
            length += sqrt(dx * dx + dy * dy) / Radius;
            endNailIndex = startNailIndex;
        }
        if (iteration < nailsOrder.size())nailsOrder[iteration] = startNailIndex;
        else nailsOrder.push_back(startNailIndex);
        updateImage(stringArt, optimalLine);
        if (iteration % (IterationNumber / 100) == 0)cout << "progress: " << 100 * iteration / IterationNumber << " %" << '\n';
        if (deadEndNails == NailsNumber) {
            cout << "All nails are dead end\n";
            break;
        }
        iteration++;
    }
    
    //prepare outpur array
    int index = nailsOrder.size() - 1;
    while (nailsOrder[index] == -1)index--;
    while (dist(nailsOrder[index], nailsOrder[index - 1], NailsNumber) < 2) {
        nailsOrder[index] = -1;
        index--;
    }
    ofstream file("result.txt"); 
    cout << "output array:\n";
    for (int i = 0; i < nailsOrder.size(); i++)
        if (nailsOrder[i] != -1) {
        cout << nailsOrder[i] << ' ';
        file << nailsOrder[i] << ' ';
        if ((i + 1) % 10 == 0) {
            cout << '\n';
            file << '\n';
        }
    }
    file.close();
    cout << "length: " << length << "\niteratoin: " << iteration << '\n';
    // View result
    Mat FinalResult(ImageHeight,ImageWidth, CV_8UC1);
    viewMatrix(stringArt, FinalResult);
    imwrite("FinalResult.jpg", FinalResult);
    imshow("result", FinalResult);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

void viewMatrix(vector <vector <uchar>> &image, Mat &output) {
    int height = image.size();
    int width = image[0].size();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output.at<uchar>(i, j) = image[i][j];
        }
    }
}

void linePosition(Pixel Pixel1, Pixel Pixel2, vector <Pixel>& line) {
    line.resize(abs(Pixel1.x - Pixel2.x) + abs(Pixel1.y - Pixel2.y) + 2);
    int index = 0;
    // 255 - black, 0 - while (inverse color)
    uchar color = 60;
    // 1 - pixel line only coordinates
    if (Pixel1.x != Pixel2.x) {
        double k = double(Pixel1.y - Pixel2.y) / (Pixel1.x - Pixel2.x);
        double b = Pixel2.y - k * Pixel2.x;
        for (int i = min(Pixel1.x, Pixel2.x); i <= max(Pixel1.x, Pixel2.x); i++) {
            line[index] = Pixel(i, int(i * k + b), color);
            index++;
        }
    }
    if (Pixel1.y != Pixel2.y) {
        double k = double(Pixel1.x - Pixel2.x) / (Pixel1.y - Pixel2.y);
        double b = Pixel2.x - k * Pixel2.y;
        for (int i = min(Pixel1.y, Pixel2.y); i <= max(Pixel1.y, Pixel2.y); i++) {
            line[index] = Pixel(int(i * k + b), i, color);
            index++;
        }
    }
    // for (int i = 0; i < line.size(); i++)cout << line[i].x << ' ' << line[i].y << '\n';
}

void updateImage(vector<vector<uchar>> &Image,const vector <Pixel> &changes) {
    for (int i = 0; i < changes.size(); i++) {
        uchar color = Image[changes[i].y][changes[i].x];
        Image[changes[i].y][changes[i].x] = max(color - changes[i].color, 0);
    }
}

vector<Pixel> createCircle(int x, int y, int R, uchar color) {
    vector<Pixel> circle((2 * R - 1) * (2 * R - 1));
    int index = 0;
    for (int i = x - R + 1; i < x + R; i++) {
        for (int j = y - R + 1; j < y + R; j++) {
            if (((x - i) * (x - i) + (y - j) * (y - j)) <= (R * R)) {
                circle[index] = Pixel(i, j, color);
                index++;
            }
            else  circle[index] = Pixel(i, j, 0);
        }
    }
    return circle;
}

double difference(const vector<Pixel> &line, const vector<vector<uchar>> &OriginalImage, const vector<vector<uchar>> &NewImage, const vector<vector<double>>& Scale) {
    double result = 0;
    bool flag = Scale.empty();
    for (int i = 0; i < line.size(); i++) {
        int diff = (OriginalImage[line[i].y][line[i].x] - NewImage[line[i].y][line[i].x]);
        int diff2 = OriginalImage[line[i].y][line[i].x] - max(NewImage[line[i].y][line[i].x] - line[i].color, 0);
        if (diff2 < 0)diff2 /= 2; // fine reduction for more black pixels
        double delta = (diff * diff) - (diff2 * diff2);
        if (!flag) {
            delta *= Scale[line[i].y][line[i].x];
        }
        result += delta;
    }

    result /= line.size();
    return result;
}

// test 
void onMouse(int event, int x, int y, int flags, void* userdata) {
    cv::Mat* image = static_cast<cv::Mat*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        // Start drawing
        circle(*image, cv::Point(x, y), 5, cv::Scalar(0, 0, 200), -1);
    }
    else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
        // Continue drawing while left button is pressed and mouse is moving
        circle(*image, cv::Point(x, y), 5, cv::Scalar(0, 0, 200), -1);
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        // Stop drawing
    }

    imshow("Distribution", *image);
}

void stepDistribution(vector<vector<double>>& scaleMatrix, double step, int R, bool Normalize) {
    int ImageHeight = scaleMatrix.size();
    int ImageWidth = scaleMatrix[0].size();
    if(R == -1)R = min(ImageHeight, ImageWidth) / 4;
    int x = ImageHeight / 2;
    int y = ImageWidth / 2;
    for (int i = x - R + 1; i < x + R; i++)
        for (int j = y - R + 1; j < y + R; j++)
            if ((x - i) * (x - i) + (y - j) * (y - j) <= R * R)
                scaleMatrix[i][j] += step;
    if(Normalize)normalize(scaleMatrix);
}

void normalize(vector<vector<double>>& scaleMatrix) {
    int ImageHeight = scaleMatrix.size();
    int ImageWidth = scaleMatrix[0].size();
    double sum = 0;
    for (int i = 0; i < ImageHeight; i++)
        for (int j = 0; j < ImageWidth; j++)
            sum += scaleMatrix[i][j];
    double baseScale = 1.0 - sum / (ImageHeight * ImageWidth);
    for (int i = 0; i < ImageHeight; i++)
        for (int j = 0; j < ImageWidth; j++)
            scaleMatrix[i][j] += baseScale;
    //check distriburion
    //print error of distribution
    sum = 0;
    for (int i = 0; i < ImageHeight; i++)
        for (int j = 0; j < ImageWidth; j++)
            sum += scaleMatrix[i][j];
    double error = 1 - sum / (ImageHeight * ImageWidth);
    cout << "Disrtibution error: " << error << '\n';
}
