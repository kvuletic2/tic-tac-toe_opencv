#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

void imgPreprocessing(Mat src, Mat& dst); //applies adaptive thresholding and other necessary preparations to the original image
void isolateGrid(Mat& src, Mat& dst); //determins which elements the grid is and removes all other elements from the image
void findCells(Mat src, Mat& dst); //splits the grid into 9 cells and displays the state of each of them

int main()
{
    std::string imgpath = SOURCE_PATH "image7.jpeg";
    Mat image = imread(imgpath);

    Mat imgBinary;
    imgPreprocessing(image, imgBinary);

    Mat grid = imgBinary.clone();
    cvtColor(grid, grid, COLOR_GRAY2BGR);
    isolateGrid(imgBinary, grid);

    findCells(imgBinary, grid);
    
    imshow("image", image);
    //imshow("binary", imgBinary);
    imshow("grid", grid);
    
    waitKey(0);
    return 0;
}

void imgPreprocessing(Mat src, Mat& dst)
{
    Mat imgGray;

    cvtColor(src, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgGray, Size(7, 7), 5, 0);
    adaptiveThreshold(imgGray, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 53, 3);
    copyMakeBorder(dst, dst, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(0, 0, 0));
    dilate(dst, dst, getStructuringElement(0, Size(3, 3), Point(1, 1)));
    erode(dst, dst, getStructuringElement(0, Size(3, 3), Point(1, 1)));
}

void isolateGrid(Mat& src, Mat& dst)
{
    std::vector<std::vector<cv::Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    int largestArea = 0; int secondLargest = 0; int index = 0; int index2 = 0; double area;
    Mat boxPts;
    double angle = 0;
    //find the largest and second largest contour
    for (size_t j = 0; j < contours.size(); j++)
    {
        area = contourArea(contours[j]);

        if (area > largestArea)
        {
            largestArea = area;
            index = j;
        }
    }

    for (size_t j = 0; j < contours.size(); j++)
    {
        area = contourArea(contours[j]);

            if (area > secondLargest && area < largestArea * 0.8)
            {
                secondLargest = area;
                index2 = j;
            }
    }
    //find the minimal area rectangle of the second largest contour to determine the rotation of the image
    std::vector<RotatedRect> minRect(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        minRect[i] = minAreaRect((contours[i]));
        
        if (i != index)
        {
            drawContours(dst, contours, (int)i, Scalar(0, 0, 0), FILLED);
        }

        if (i == index2)
        {
            Point2f boxPts[4];
            minRect[i].points(boxPts);

            if (abs(boxPts[0].x - boxPts[1].x) > abs(boxPts[0].y - boxPts[1].y))
            {
                angle = atan(abs(boxPts[0].y - boxPts[1].y) / abs(boxPts[0].x - boxPts[1].x));
            }

            else
            {
                angle = atan(abs(boxPts[0].x - boxPts[1].x) / abs(boxPts[0].y - boxPts[1].y));
            }

            /*for (size_t k = 0; k < 4; k++)
            {
                line(dst, boxPts[k], boxPts[(k + 1) % 4], Scalar(0, 255, 0));
            }*/
        }
    }

    angle = angle * 180 / 3.14;
    Point2f center = Point(src.cols / 2, src.rows / 2);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);

    warpAffine(src, src, rot, dst.size());
    warpAffine(dst, dst, rot, dst.size());
}

void findCells(Mat src, Mat& dst)
{
    Mat temp = src.clone();
    std::vector<std::vector<cv::Point>> contours2;
    std::vector<Vec4i> hierarchy2;

    findContours(temp, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE);

    int largestArea2 = 0; int secondLargest = 0; int index = 0; int index2 = 0; double area;

    for (size_t j = 0; j < contours2.size(); j++)
    {
        area = contourArea(contours2[j]);

        if (area > largestArea2)
        {
            largestArea2 = area;
            index = j;
        }
    }

    for (size_t j = 0; j < contours2.size(); j++)
    {
        area = contourArea(contours2[j]);

        if (area > secondLargest && area < largestArea2 * 0.8)
        {
            secondLargest = area;
            index2 = j;
        }
    }

    Rect bounding1 = boundingRect(contours2[index]);
    Rect bounding2 = boundingRect(contours2[index2]);

    //find the distance between the largest contour and the grid center
    Point offset((bounding2.x + bounding2.width / 2) - (bounding1.x + bounding1.width / 2), (bounding2.y + bounding2.height / 2) - (bounding1.y + bounding1.height / 2));

    int largestArea = 0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<Vec4i> hierarchy;

    //offset contours to the grid center and draw the bounding rectangle of the largest contour
    findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, offset);

    for (size_t j = 0; j < contours.size(); j++)
    {
        area = contourArea(contours[j]);

        if (area > largestArea)
        {
            largestArea = area;
            index = j;
        }
    }
    
    Rect bounding = boundingRect(contours[index]);
    bounding.x += bounding.width * 0.05;
    bounding.y += bounding.height * 0.05;
    bounding.width = bounding.width * 0.9;
    bounding.height = bounding.height * 0.9;
    //rectangle(dst, bounding, Scalar(0, 255, 0), 2, LINE_8);

    float cellHeight = (float)bounding.height / 3;
    float cellWidth = (float)bounding.width / 3;
    Mat temp2 = src.clone();
    cvtColor(temp2, temp2, COLOR_GRAY2BGR);

    std::vector<std::vector<cv::Point>> contours1;
    std::vector<Vec4i> hierarchy1;
    findContours(src, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_SIMPLE);
    largestArea = 0;  index = 0;

    for (size_t j = 0; j < contours1.size(); j++)
    {
        area = contourArea(contours1[j]);

        if (area > largestArea)
        {
            largestArea = area;
            index = j;
        }
    }

    drawContours(src, contours1, index, Scalar(0, 0, 0), FILLED);

    secondLargest = 0; index2 = 0;
    for (size_t j = 0; j < contours2.size(); j++)
    {
        area = contourArea(contours2[j]);

        if (area > secondLargest && area < largestArea2 * 0.8)
        {
            secondLargest = area;
            index2 = j;
        }
    }
    
    int index3 = hierarchy1[index2][2];
    while (index3 != -1)
    {
        if (contourArea(contours1[index3]) > 100)
        {
            drawContours(src, contours1, index3, Scalar(255, 255, 255), FILLED);
            if (hierarchy1[index3][2] > -1)
            {
                drawContours(src, contours1, hierarchy1[index3][2], Scalar(0, 0, 0), FILLED);
            }
            break;
        }
        else
        {
            index3 = hierarchy1[index3][0];
        }
    }

    for (size_t y = 0; y < 3; y++)
    {
        for (size_t x = 0; x < 3; x++)
        {
            //split the grid into 9 cells
            Rect cell;
            cell.x = bounding.x + (x * cellWidth * 0.1) + (x * cellWidth);
            cell.y = bounding.y + (y * cellHeight * 0.1) + (y * cellHeight);
            cell.width = cellWidth * 0.8;
            cell.height = cellHeight * 0.8;

            Rect cell2;
            cell2.x = bounding.x + (x * cellWidth);
            cell2.y = bounding.y + (y * cellHeight);
            cell2.width = cellWidth;
            cell2.height = cellHeight;

            rectangle(temp2, cell2, Scalar(0, 255, 0), 2, LINE_8);
            rectangle(src, cell2, Scalar(0, 0, 0), 2, LINE_8);
            rectangle(dst, cell, Scalar(0, 255, 0), 2, LINE_8);

            Mat isolatedCell = src(cell2);

            //determine if there is a circle in each cell
            std::vector<Vec3f> circles;
            HoughCircles(isolatedCell, circles, HOUGH_GRADIENT, 1, isolatedCell.rows / 16, 50, 20);
            for (size_t p = 0; p < circles.size(); p++)
            {
                Point center(cvRound(circles[p][0]) + cell2.x, cvRound(circles[p][1]) + cell2.y);
                int radius = cvRound(circles[p][2]);
                // draw the circle center
                circle(temp2, center, 3, Scalar(0, 255, 0), -1, 8, 0);
                // draw the circle outline
                circle(temp2, center, radius, Scalar(0, 0, 255), 3, 8, 0);
            }

            bool circleFound = false;
            for (size_t k = 0; k < circles.size(); k++)
            {
                if (circles[k][2] > 15)
                {
                    circleFound = true;
                    break;
                }
            }

            if (circleFound)
            {
                circle(dst, Point(cell.x + (cell.width / 2), cell.y + (cell.height / 2)), cell.width / 2.0 * 0.8, Scalar(0, 0, 255), 3);
            }

            //if there is no circle determine if the cell contains an X or if it's empty
            else
            {
                std::vector<std::vector<cv::Point>> contours3;
                std::vector<Vec4i> hierarchy3;

                findContours(isolatedCell, contours3, hierarchy3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                if (contours3.size() > 0)
                {
                    int largestArea4 = 0; int index4 = 0; double area4;

                    for (size_t j = 0; j < contours3.size(); j++)
                    {
                        area4 = contourArea(contours3[j]);

                        if (area4 > largestArea4)
                        {
                            largestArea4 = area4;
                            index4 = j;
                        }
                    }

                    std::vector<std::vector<Point>> hull(contours3.size());

                    convexHull(contours3[index4], hull[index4]);
                    if (contourArea(hull[index4]) > 400)
                    {
                        line(dst, Point(cell.x + (cell.width * 0.1), cell.y + (cell.height * 0.1)), Point(cell.x + (cell.width * 0.9), cell.y + (cell.height * 0.9)), Scalar(255, 0, 0), 3);
                        line(dst, Point(cell.x + (cell.width * 0.1), cell.y + (cell.height * 0.9)), Point(cell.x + (cell.width * 0.9), cell.y + (cell.height * 0.1)), Scalar(255, 0, 0), 3);
                    }
                }
            }
        }
    }
    //imshow("temp2", temp2);
    //waitKey(0);
}