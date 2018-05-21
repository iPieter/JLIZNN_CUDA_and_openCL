#include <stdio.h>
#include <stdlib.h>
#include <QGraphicsScene>
#include <QTextStream>
#include <QTimer>
#include <QPixmap>
#include <QSignalMapper>
#include <QFileDialog>
#include <QDrag>
#include <QMimeData>
#include <QTemporaryFile>

#include <algorithm>    // std::max

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define TEST_INDEX(i,size) ((i >= 0 && i < size) ? true : false)

#ifndef STBI_INCLUDE_STB_IMAGE_H
#include "std_image.h"
#include "std_image_write.h"
#endif

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "pipeline.cpp"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //connect(ui->open, SIGNAL(QAction::toggled()), this, SLOT(MainWindow::on_open_triggered));
    //show openCL devices
    printDevices();

    //centralWidget()->setAttribute(Qt::WA_TransparentForMouseEvents);
    //setMouseTracking(true);

}

MainWindow::~MainWindow()
{
    if (temp_file != NULL )
    {
        QFile file (temp_file);
        file.remove();
    }

    delete ui;
}

void MainWindow::on_pushButton_pressed()
{
    //ui->setupUi(this);

}

void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    QTextStream(stdout) << "slider moved to " << position << '\n';
}

void MainWindow::resize()
{
    QTextStream(stdout) << "resizing\n";
    ui->mainImage->fitInView(scene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    if (img != NULL)
        resize();

}

void MainWindow::on_pushButton_2_pressed()
{
    QTextStream(stdout) << "kernel\n";

    int i;
    for( i = 0; i < w * h * comp; i += comp)
    {
        img[i + 1] = img[i];
        img[i + 2] = img[i];
    }

    QImage imageQ(img, w, h, comp == 3 ? QImage::Format_RGB888 : QImage::Format_RGBA8888);
    //image.load("/Users/Pieter/Desktop/Schermafbeelding 2018-02-10 om 13.14.20.png");
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());
}

void MainWindow::on_pushButton_3_pressed()
{
    unsigned char* img_original( new unsigned char[ w * h * comp]);
    memcpy( img_original, img, w*h*comp*sizeof(unsigned char) );

    /*
    int **gKernel;
    gKernel = new int *[kernel_size];
    for(int i = 0; i <kernel_size; i++)
        gKernel[i] = new int[kernel_size];
    int sum = createFilter(gKernel, kernel_size);

    for (int x = 0; x < kernel_size; ++x)
    {
        QTextStream(stdout) << "{";

        for (int y = 0; y < kernel_size; ++y)
        {
            QTextStream(stdout) << gKernel[x][y] << ", ";
        }
        QTextStream(stdout) << "}\n";

    }

    QTextStream(stdout) << "sum" << sum << "\n";


    for( int x = 0; x < w; x++ )
    {
        for( int y = 0; y < h; y++ )
        {
            if( x + y * w < w*h * 3 )
            {
                int sum_r_x = 0;
                int sum_g_x = 0;
                int sum_b_x = 0;

                for( int i = 0; i < kernel_size; i++ )
                {
                    for( int j = 0; j < kernel_size; j++ )
                    {
                        int index = (x + i - kernel_size / 2 + (y + j - kernel_size /2) * w) * 3;
                        if( TEST_INDEX(index, w*h * 3))
                        {
                            sum_r_x += gKernel[i][j] * img_original[index];
                            sum_g_x += gKernel[i][j] * img_original[index + 1];
                            sum_b_x += gKernel[i][j] * img_original[index + 2];
                        }
                    }
                }

                img[ (x + y * w) * 3 + 0 ] = (unsigned char)(sum_r_x / sum / sum );
                img[ (x + y * w) * 3 + 1 ] = (unsigned char)(sum_g_x / sum / sum );
                img[ (x + y * w) * 3 + 2 ] = (unsigned char)(sum_b_x / sum / sum );
            }
        }
    }
    */
    for ( auto device : enabled_devices )
        run( img_original, img, w, h, comp, device[0], device[1], kernel_size);

    delete scene;

    temp_file == NULL;

    QImage imageQ(img, w, h, comp == 3 ? QImage::Format_RGB888 : QImage::Format_RGBA8888);
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());

    delete [ ] img_original;
    img_original = NULL;

    //for(int i = 0; i < kernel_size; i++)
        //delete [ ] gKernel[i];
    //delete [ ] gKernel;
    //gKernel = NULL;
}


void MainWindow::createFilter(double **gKernel, int size)
{
    // set standard deviation
    // https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
    double sigma = ( size -1 ) / 2;

    // sum is for normalization
    double sum = 0.0;

    if (size % 2 == 0)
        throw "uneven numbers.";

    double mean = size/2;

    // generate kernel
    for (int x = 0; x < size; ++x)
        for (int y = 0; y < size; ++y) {
            gKernel[x][y] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                             / (2 * M_PI * sigma * sigma);

            // Accumulate the kernel values
            sum += gKernel[x][y];
        }

    // Normalize the kernel
    for (int x = 0; x < size; ++x)
        for (int y = 0; y < size; ++y)
            gKernel[x][y] /= sum;

}

int MainWindow::createFilter(int **gKernel, int size)
{
    //generate pascals triangle
    int *row = new int[size];

    size--;

    row[0] = 1; //First element is always 1
    int sum = 1;
    for(int i=1; i<size/2+1; i++){ //Progress up, until reaching the middle value
        row[i] = row[i-1] * (size-i+1)/i;
        sum += row[i];
    }
    for(int i=size/2+1; i<=size; i++){ //Copy the inverse of the first part
        row[i] = row[size-i];
        sum += row[i];
    }

    size++;

    // generate kernel
    for (int x = 0; x < size; ++x)
        for (int y = 0; y < size; ++y)
            gKernel[x][y] = row[x] * row[y];

    delete [ ] row;
    row = NULL;

    return sum;
}

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    kernel_size = value % 2 ? value : value + 1;
}

void MainWindow::on_pushButton_4_pressed()
{
    QTextStream(stdout) << "colour grading\n";

    int i;
    for( i = 0; i < w * h * comp; i += comp)
    {
        img[i + 0] = std::min( 255, std::max( 0, (int) ( 255.0 * img[i + 0] / r ) ) );
        img[i + 1] = std::min( 255, std::max( 0, (int) (  255.0 * img[i + 1] / g ) ) );
        img[i + 2] = std::min( 255, std::max( 0, (int) (  255.0 * img[i + 2] / b ) ) );
    }

    QImage imageQ(img, w, h, comp == 3 ? QImage::Format_RGB888 : QImage::Format_RGBA8888);
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());
}

void MainWindow::on_horizontalSlider_4_valueChanged(int value)
{
    QTextStream(stdout) << "setting r\n";
    r = value;
}

void MainWindow::on_horizontalSlider_3_valueChanged(int value)
{
    QTextStream(stdout) << "setting g\n";
    g = value;
}

void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    QTextStream(stdout) << "setting b\n";
    b = value;
}

int MainWindow::printDevices() {

    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_ulong maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            QAction* newAct = new QAction(tr(value), this);
            newAct->setCheckable(true);
            //newAct->setShortcuts(QKeySequence::New);
            newAct->setStatusTip(tr("Create a new file"));

            connect(newAct, &QAction::toggled, this, [this, newAct, i, j](){
                this->toggleDevice(i, j, newAct->isChecked());
            });

            std::string shortcut("Ctrl+" + std::to_string(i+j));
            printf("%d. SHORTCUT: %s\n", j+1, shortcut.c_str());
            newAct->setShortcut(QApplication::translate("MainWindow", shortcut.c_str() , nullptr));

            ui->menuWindow->addAction(newAct);


        }

        free(devices);

    }

    free(platforms);

    return 0;

}

void MainWindow::toggleDevice(int platform, int device, bool toggle)
{
    if ( toggle )
    {
        enabled_devices.push_back(new int[2]{ platform, device });
    }
    else
    {
        enabled_devices.erase(
           std::remove_if(
              enabled_devices.begin(),
              enabled_devices.end(),
              [platform, device]( const int* v ){ return v[0] == platform && v[1] == device; }
          ),
          enabled_devices.end()
        );
    }

    for( int* n : enabled_devices )
    {
       QTextStream(stdout) << "Device: " << n[0] << ", " << n[1] << '\n';
    }
}

void MainWindow::on_open_triggered()
{
    if (img != NULL)
    {
        delete[] img;
        delete scene;
        //delete imageQ;
    }

    auto filename = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "~", tr("Image Files (*.png *.jpg *.bmp)"));

    img = stbi_load( filename.toUtf8().constData(), &w, &h, &comp, STBI_default );

    cl_uint platform_id_count = 0;
    clGetPlatformIDs( 0, nullptr, &platform_id_count );

    if(img == NULL)
    {
        QTextStream(stdout) << "Couldn't load image\n";
    }

    QTextStream(stdout) << "Loaded img: " << w << ", " << h << ", " << comp << '\n';

    QImage imageQ(img, w, h, comp == 3 ? QImage::Format_RGB888 : QImage::Format_RGBA8888);
    //image.load("/Users/Pieter/Desktop/Schermafbeelding 2018-02-10 om 13.14.20.png");
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());

    QTimer::singleShot(200, this, SLOT(resize()));
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        dragStartPosition = event->pos();
        QTextStream(stdout) << dragStartPosition.x() <<"\n";

    }

}
void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    if (!(event->buttons() & Qt::LeftButton))
        return;
    if ((event->pos() - dragStartPosition).manhattanLength() < QApplication::startDragDistance())
        return;
    if (event->pos().x() > ui->mainImage->geometry().topLeft().x() &&
        event->pos().x() < ui->mainImage->geometry().bottomRight().x() &&
        event->pos().y() > ui->mainImage->geometry().topLeft().y() &&
        event->pos().y() < ui->mainImage->geometry().bottomRight().y())
    {
        if ( temp_file == NULL )
        {
            QTemporaryFile file;
            file.setFileTemplate("XXXXXX.png");
            file.setAutoRemove(false);
            file.open();
            temp_file = file.fileName();
            stbi_write_png(temp_file.toUtf8().constData(), w, h, comp, (void *)img, 0 );
        }

        QImage imageQ(img, w, h, comp == 3 ? QImage::Format_RGB888 : QImage::Format_RGBA8888);

        QUrl url = QUrl::fromLocalFile(temp_file);

        QDrag* drag = new QDrag(this);
        QMimeData* mimeData = new QMimeData;
        mimeData->setUrls(QList<QUrl>() << url);
        QTextStream(stdout) <<temp_file <<"\n";

        mimeData->setImageData(imageQ);
        drag->setMimeData(mimeData);

        Qt::DropAction dropAction = drag->exec(Qt::CopyAction | Qt::MoveAction);
    }
}
