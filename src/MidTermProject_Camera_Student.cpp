/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <string>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{   

    std::ofstream myfile;
    myfile.open ("Results.csv");

    vector<string> detectList = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", 
                                                                "AKAZE", "SIFT"};
    vector<string> descList   = {"BRISK", "BRIEF", "ORB", "FREAK",  "SIFT"};

    for (string detectorType : detectList)
    {

    for (string descriptorType : descList)
    {

    if (detectorType.compare("SIFT") == 0 && descriptorType.compare("ORB") == 0)
        continue;

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // string detectorType = "SIFT"; //// SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    // string descriptorType = "AKAZE"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

    vector<double> detectorTTClist, descTTClist;
    vector<int> kptsList, mtchList;

    myfile << detectorType << " + "<< descriptorType << "\n";
    myfile << " , Image 1, Image 2, Image 3, Image 4, Image 5, Image 6, Image 7, Image 8, Image 9.\n";

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        if (dataBuffer.size() == dataBufferSize)
            dataBuffer.erase(dataBuffer.begin());

        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        double tDetect, tDescrb;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {   
            tDetect = (double)cv::getTickCount();
            detKeypointsShiTomasi(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsHarris(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("FAST") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsFAST(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("BRISK") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsBRISK(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("ORB") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsORB(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("AKAZE") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsAKAZE(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        else if (detectorType.compare("SIFT") == 0)
        {
            tDetect = (double)cv::getTickCount();
            detKeypointsSIFT(keypoints, imgGray, false);
            tDetect = ((double)cv::getTickCount() - tDetect) / cv::getTickFrequency();
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for (auto it = keypoints.begin(); it != keypoints.end(); )
            {
                if ((it->pt.x <  vehicleRect.x)                       || 
                    (it->pt.x > (vehicleRect.x + vehicleRect.width))  ||
                    (it->pt.y <  vehicleRect.y)                       ||
                    (it->pt.y > (vehicleRect.y + vehicleRect.height))   )
                {
                    keypoints.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            if (descriptorType.compare("SIFT") == 0)
                matcherType = "MAT_FLANN";


            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            tDescrb = (double)cv::getTickCount();

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            tDescrb = ((double)cv::getTickCount() - tDescrb) / cv::getTickFrequency();

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                // cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;

        }


        if (dataBuffer.size() == dataBufferSize)
        {
            cout << "-----------------------------------" << endl;
            cout << detectorType   << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * tDetect / 1.0 << " ms" << endl;
            cout << descriptorType << " description with m=" << (dataBuffer.end() - 1)->kptMatches.size() 
                                            << " matches in " << 1000 * tDescrb / 1.0 << " ms" << endl;
            cout << "KeyPoints: "  << keypoints.size()                         << endl;
            cout << "Matches: "    << (dataBuffer.end() - 1)->kptMatches.size()    << endl;
            cout << "-----------------------------------" << endl;

            detectorTTClist.push_back(1000 * tDetect / 1.0);
            descTTClist.push_back(1000 * tDescrb / 1.0);
            kptsList.push_back(keypoints.size());
            mtchList.push_back((dataBuffer.end() - 1)->kptMatches.size() );
        }

    } // eof loop over all images

    myfile << "Detection Time, ";
    for (double i : detectorTTClist)
    {
        myfile << std::to_string(i) << ",";
    }
    myfile << "\n";
    myfile << "Description Time, ";
    for (double i : descTTClist)
    {
        myfile << std::to_string(i) << ",";
    }
    myfile << "\n";
    myfile << "Keypoints, ";
    for (int i : kptsList)
    {
        myfile << std::to_string(i) << ",";
    }
    myfile << "\n";
    myfile << "Matches, ";
    for (int i : mtchList)
    {
        myfile << std::to_string(i) << ",";
    }
    myfile << "\n";
    myfile << "\n";
    myfile << "\n";

    }
    }

    return 0;
}
