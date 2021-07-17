/* -- Serge Bobbia : serge.bobbia@u-bourgogne.fr -- Le2i 2018
 * This work is distributed for non commercial use only,
 * it implements the IBIS method as described in the ICPR 2018 paper.
 * Read the ibis.h file for options and benchmark instructions
 *
 * This file show how to instanciate the IBIS class
 * You can either provide a file, or a directory, path to segment images
 */

#include <chrono>
using namespace std::chrono;
#include <iostream>
#include "ibis.h"
#include <opencv2/opencv.hpp>
// #include "opencv-3.4.1/include/opencv2/opencv.hpp"
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "utils.h"
// #include <H5File.h>
#include "signal_processing.h"
#include <filesystem>
namespace fs = std::__fs::filesystem;

#define SAVE_output         1
#define visu                1
#define visu_SNR	    	0
#define signal_size         300
#define signal_processing   0

using namespace std;
//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void DrawContoursAroundSegments(
    unsigned char*&			ubuff,
    int*&					labels,
    const int&				width,
    const int&				height,
    const unsigned int&				color )
{
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    int sz = width*height;
    vector<bool> istaken(sz, false);
    vector<int> contourx(sz);
    vector<int> contoury(sz);
    int mainindex(0);int cind(0);

    for( int j = 0; j < height; j++ )
    {
        for( int k = 0; k < width; k++ )
        {
            int np(0);
            for( int i = 0; i < 8; i++ )
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                    int index = y*width + x;

                    //if( false == istaken[index] )//comment this to obtain internal contours
                    {
                        if( labels[mainindex] != labels[index] ) np++;
                    }
                }
            }
            if( np > 1 )
            {
                contourx[cind] = k;
                contoury[cind] = j;
                istaken[mainindex] = true;
                //img[mainindex] = color;
                cind++;
            }
            mainindex++;
        }
    }

    int numboundpix = cind;//int(contourx.size());
    for( int j = 0; j < numboundpix; j++ )
    {
        int ii = contoury[j]*width + contourx[j];
        ubuff[ii] = 0xff;

        for( int n = 0; n < 8; n++ )
        {
            int x = contourx[j] + dx8[n];
            int y = contoury[j] + dy8[n];
            if( (x >= 0 && x < width) && (y >= 0 && y < height) )
            {
                int ind = y*width + x;
                if(!istaken[ind])
                    ubuff[ind] = 0;
            }
        }
    }
}


void write_labels(IplImage* input, const std::string& output_labels)
{
    std::ofstream file;
    file.open(output_labels.c_str());

    unsigned char* data = (unsigned char*)input->imageData;
    for (int y=0 ; y<input->height ; y++)
    {
        for (int x=0 ; x<input->width-1 ; x++)
        {
            file << (int) data[y*input->widthStep + x*input->nChannels] << " ";

        }
        file << (int) data[y*input->widthStep + (input->width -1)*input->nChannels] << std::endl;

    }

    file.close();
}

void write_traces(float* C1, float* C2, float* C3, const std::string& output_labels, IBIS* SP)
{
    std::ofstream file;
    file.open(output_labels.c_str());
    int max_sp = SP->getMaxSPNumber();

    for (int y=0 ; y<SP->getActualSPNumber() ; y++)
    {
        file << (double) C1[y] << ",";
        file << (double) C2[y] << ",";
        file << (double) C3[y] << std::endl;

    }

    file.close();
}

void execute_IBIS( int K, int compa, IBIS* Super_Pixel, Signal_processing* Signal, cv::Mat* img, std::string output_basename, int frame_index ) {

    int width = img->cols;
    int height = img->rows;
    int size = width * height;

    // process IBIS
    Super_Pixel->process( img );

    int* labels = Super_Pixel->getLabels();

    cv::Mat* output_bounds = new cv::Mat(cvSize(width, height), CV_8UC1);
    const int color = 0xFFFFFFFF;

    unsigned char* ubuff = output_bounds->ptr();
    std::fill(ubuff, ubuff + (width*height), 0);

    DrawContoursAroundSegments(ubuff, labels, width, height, color);

    cv::Mat* pImg = new cv::Mat(cvSize(width, height), CV_8UC3);
    float* sum_rgb = new float[Super_Pixel->getMaxSPNumber()*3];
    int* count_px = new int[Super_Pixel->getMaxSPNumber()];
    std::fill(sum_rgb, sum_rgb+Super_Pixel->getMaxSPNumber()*3, 0.f);
    std::fill(count_px, count_px+Super_Pixel->getMaxSPNumber(), 0);

    int ii = 0, i;
    for (i = 0; i < 3 * size; i += 3, ii++) {
        count_px[ labels[ii] ]++;
        sum_rgb[ labels[ii] + Super_Pixel->getMaxSPNumber() * 0 ] += img->ptr()[i];
        sum_rgb[ labels[ii] + Super_Pixel->getMaxSPNumber() * 1 ] += img->ptr()[i+1];
        sum_rgb[ labels[ii] + Super_Pixel->getMaxSPNumber() * 2 ] += img->ptr()[i+2];

    }

    float* R = new float[Super_Pixel->getMaxSPNumber()];
    float* G = new float[Super_Pixel->getMaxSPNumber()];
    float* B = new float[Super_Pixel->getMaxSPNumber()];

    float* R_avg = new float[Super_Pixel->getMaxSPNumber()];
    float* G_avg = new float[Super_Pixel->getMaxSPNumber()];
    float* B_avg = new float[Super_Pixel->getMaxSPNumber()];
    memset( R_avg, 0, sizeof(float) * Super_Pixel->getMaxSPNumber() );
    memset( G_avg, 0, sizeof(float) * Super_Pixel->getMaxSPNumber() );
    memset( B_avg, 0, sizeof(float) * Super_Pixel->getMaxSPNumber() );

    for (i=0; i<Super_Pixel->getMaxSPNumber(); i++) {
        sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 0 ] /= count_px[ i ];
        sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 1 ] /= count_px[ i ];
        sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 2 ] /= count_px[ i ];

        R[i] = sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 2 ];
        G[i] = sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 1 ];
        B[i] = sum_rgb[ i + Super_Pixel->getMaxSPNumber() * 0 ];

    }

    // increase stat
    int* adj = Super_Pixel->get_adjacent_sp();
    int* nb_adj = Super_Pixel->nb_adjacent_sp();
    float dist;
    for (i=0; i<Super_Pixel->getActualSPNumber(); i++) {
        ii=0;

        for( int j=0; j<nb_adj[i]; j++ ) {
            int sp = adj[9*i+j];

            dist = ( R[ i ] - R[ sp ] ) * ( R[ i ] - R[ sp ] ) +
                   ( G[ i ] - G[ sp ] ) * ( G[ i ] - G[ sp ] ) +
                   ( B[ i ] - B[ sp ] ) * ( B[ i ] - B[ sp ] );
            dist /= 9;

            if( dist < 100.f ) {
                R_avg[ i ] += R[ sp ];
                G_avg[ i ] += G[ sp ];
                B_avg[ i ] += B[ sp ];
                ii++;

            }

        }

	R_avg[i] = R[i]/Super_Pixel->getActualSPNumber();
	G_avg[i] = G[i]/Super_Pixel->getActualSPNumber();
	B_avg[i] = B[i]/Super_Pixel->getActualSPNumber();
    }

    // signal processing
#if signal_processing
    Signal->add_frame( Super_Pixel->get_inheritance(),
                       R_avg,
                       G_avg,
                       B_avg,
                       Super_Pixel->getActualSPNumber() );

    Signal->process();
#endif
    if( frame_index % 30 == 0 ) {
        printf("-frame\t%i\n", frame_index);

    }

#if visu
        // SNR superposition
        const float* SNR;
        if( frame_index > signal_size ) {
            SNR = Signal->get_SNR();
        }

        for (i=0, ii=0; i < 3 * size; i += 3, ii++) {
            int sp = labels[ii];

            if (sp >= 0) {

                pImg->ptr()[i + 2]  = (unsigned char) img->ptr()[i+2];
                pImg->ptr()[i + 1]  = (unsigned char) img->ptr()[i+1];
                pImg->ptr()[i]      = (unsigned char) img->ptr()[i+0];

                if( ubuff[ ii ] == 255 ) {
                    pImg->ptr()[i + 2]  = 255;
                    pImg->ptr()[i + 1]  = 255;
                    pImg->ptr()[i]      = 255;

                }

#if signal_processing
                if( frame_index > signal_size ) {
                    if( SNR[ labels[ii] ] > 0 && ubuff[ ii ] == 255 ) {
                        pImg->ptr()[i + 2]  = 0;
                        pImg->ptr()[i + 1]  = 0;
                        pImg->ptr()[i]      = 0;

                        if( SNR[ labels[ii] ] > 5 )
                            pImg->ptr()[i + 0]  = 255;
                        else
                            pImg->ptr()[i + 0]  = (unsigned char)(255 * SNR[ labels[ii] ] / 5 );

                    }

                }
#endif

            }

        }


#if signal_processing
        // add text
        char text[255] = "";
        sprintf( text, "HR: %i", Signal->get_HR() );
        cv::putText(*pImg, text, cv::Point(30,30),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200,200,250), 1, CV_AA);

#if visu_SNR
        if (frame_index > signal_size) {
        for( int i=0; i<Super_Pixel->getActualSPNumber(); i++ ) {
            sprintf( text, "%.1f", SNR[i] );

            cv::putText(*pImg, text, cv::Point( int(round(double(Super_Pixel->get_Xseeds()[i]))), int(round(double(Super_Pixel->get_Yseeds()[i]))) ),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(0,0,250), 1, CV_AA);


        }
}
#endif
#endif

        cv::imshow("rgb mean", *pImg);
        cv::waitKey( 1 );

    //}
#endif

#if SAVE_output
    char output_labels[255] = {0};
    sprintf(output_labels, "results/%s/traces_%04i.csv", output_basename.c_str(), frame_index);
    write_traces( R, G, B, output_labels, Super_Pixel );

    sprintf(output_labels, "results/%s/parent_%04i.seg", output_basename.c_str(), frame_index);

    std::ofstream file;
    file.open(output_labels);
    int* parent = Super_Pixel->get_inheritance();

    for (int y=0; y<Super_Pixel->getActualSPNumber(); y++)
        file << parent[y] << std::endl;

    file.close();

#endif
    delete pImg;
    delete output_bounds;
    delete[] sum_rgb;
    delete[] count_px;
    delete[] R;
    delete[] G;
    delete[] B;
    delete[] R_avg;
    delete[] G_avg;
    delete[] B_avg;

}

int filter( const struct dirent *name ) {
    std::string file_name = std::string( name->d_name );
    std::size_t found = file_name.find(".avi");
    if (found!=std::string::npos) {
        return 1;

    }

    return 0;

}

// int get_sp_labels( int argc, char* argv[] )
int get_sp_labels( int K, int compa, const char* video_path, const char* save_path, std::string save_name)
{
    printf(" - Temporal IBIS - \n\n");

    if( K < 0 || compa < 0 ) {
        printf("--> usage ./IBIS_temporal SP_number Compacity File_path\n");
        printf(" |-> SP_number: user fixed number of superpixels, > 0\n");
        printf(" |-> Compacity: factor of caompacity, set to 20 for benchmark, > 0\n");
        printf(" |-> File_path: path to the file or device to use\n");
        printf("\n");
        printf("--> output files are saved in a \"./results\" directory\n");

        exit(EXIT_SUCCESS);
    }

    // determine mode : file or path
    struct stat sb;

    if (stat(video_path, &sb) == -1) {
        perror("stat");
        exit(EXIT_SUCCESS);
    }

    int type;
    //printf("file type : %i\n", sb.st_mode & S_IFMT);

    switch (sb.st_mode & S_IFMT) {
    case S_IFDIR:
        printf("directory processing\n");
        type=0;
        break;

    case S_IFREG:
        printf("single file processing\n");
        type=1;
        break;

    case 8192:
        printf("Device video processing\n");
        type=2;
        break;

    default:
        type=-1;
        break;

    }

    if( type == -1 )
        exit(EXIT_SUCCESS);
    else if(type == 0){

        IBIS Super_Pixel( K, compa );
        Signal_processing Signal( K, signal_size );
        
        cv::Mat img;
        int ii=0;
        std::string output_basename = std::string(save_path)+save_name;
        // cout << output_basename << endl;

        char command[255] = {0};
        sprintf( command, "mkdir -p results/%s\n", output_basename.c_str() );
        system( command );
        
        // cout << video_path;
        for (const auto & entry : std::__fs::filesystem::directory_iterator(video_path)) {
            std::string img_path = entry.path().string();
            if ( img_path.find(".png") != std::string::npos ){
                img = cv::imread(img_path);
                if (img.empty())
                {
                    std::cout << "!!! Failed imread(): image not found" << std::endl;
                    // don't let the execution continue, else imshow() will crash.
                }
                else{
                    execute_IBIS( K, compa, &Super_Pixel, &Signal, &img, output_basename, ii );
                    // __asm__("int $3");
                    ii++;
                }
            }
        }   
    }
    else if( type >= 1 ) {


       
        // // IBIS
        // IBIS Super_Pixel( K, compa );
        // Signal_processing Signal( K, signal_size );

        // // get picture
        // cv::VideoCapture video( video_path );
        // if(!video.isOpened()) { // check if we succeeded
        //     printf("Can't open this device or video file.\n");
        //     exit(EXIT_SUCCESS);

        // }

        // cv::Mat img;
        // int ii=0;
        // std::string output_basename = std::string(save_path)+save_name;
        // cout << output_basename << endl;
        // // return 0;

        // // std::string output_basename = std::string(save_path) + std::string(video_path.substr(24, video_path.find("."));
        // // printf(video_path.substr(24, video_path.find("."));

        // char command[255] = {0};
        // sprintf( command, "mkdir -p results/%s\n", output_basename.c_str() );
        // system( command );

        // while( video.read( img ) ) {
        //     execute_IBIS( K, compa, &Super_Pixel, &Signal, &img, output_basename, ii );
        //     ii++;

        // }

    }
    // exit(EXIT_SUCCESS);
}

int dirExists(const char* const path)
{
    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? 1 : 0;
}

int main(int argc, char* argv[])
{   auto start = high_resolution_clock::now();

    int K = 1000;
    int compa = 20;
    // Path to input directory for videos
    std::string path = "/Volumes/T7/PURE_unzipped/";
    // Path to IBIS output files
    const char * save_path = "../results/delete_1000/";
    int f_count = 0;
    // for (const auto & entry : std::__fs::filesystem::recursive_directory_iterator(path))
    for (const auto & entry : std::__fs::filesystem::directory_iterator(path))
    {   
        // std::string path_string{path.u8string()}
        std::string file_name = entry.path().string();
        // cout << file_name << endl;

        std::string save_name = file_name.substr(26, 5);

        const char * video_path = entry.path().c_str();
        std::string check_filename = std::string(save_path) + save_name;

        struct stat buffer;
        char* check = &check_filename[1];
        if (stat(check, &buffer) == 0) {
            cout << check_filename.substr(19) << " already present" << endl;
            f_count++;
            continue;
        }
        else {
            get_sp_labels(K, compa, video_path, save_path, save_name);
            f_count++;
            cout << "files completed: " << f_count << endl;
        }

        // if ( file_name.find(".m4v") != std::string::npos )
        // {
        //     if ( file_name.find("._") != std::string::npos )
        //     {
        //         continue;
        //     }
        //     // cout << file_name << endl;
        //     std::string save_name = file_name.substr(32, 12);
        //     cout << save_name << endl;
        //     const char * video_path = entry.path().c_str();
        //     // get_sp_labels(K, compa, video_path, save_path, save_name);
            
        //     cout << "Processed: " << file_name << endl;
        //     // f_count++;
        //     cout << "files completed: " << f_count << endl;
        //     break;
        // }

        // f_count++;
        break;
    }
    cout << f_count << std::endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    // To get the value of duration use the count()
    // member function on the duration object
    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;
    return 0;
}