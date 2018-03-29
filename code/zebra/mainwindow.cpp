#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <QtGui>
#include <QLabel>
#include <QListWidgetItem>
#include <QSizePolicy>
#include <QListWidget>
#include <QGraphicsScene>
#include <QPushButton>
#include <QTextStream>
#include <QTimer>
#include <QPixmap>

#include <algorithm>    // std::max

#define STB_IMAGE_IMPLEMENTATION
#define TEST_INDEX(i,size) ((i >= 0 && i < size) ? true : false)

#include "std_image.h"

#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QPushButton *button = new QPushButton("");

    //button->setObjectName(QStringLiteral("btn_gaussian_blur"));
    QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    sizePolicy.setHeightForWidth(button->sizePolicy().hasHeightForWidth());
    button->setSizePolicy(sizePolicy);
    button->setMaximumSize(QSize(44, 44));
    button->setBaseSize(QSize(44, 44));

    QIcon icon;
    icon.addFile(QStringLiteral(":/icons/icons/btn_mask.png"), QSize(), QIcon::Normal, QIcon::Off);
    button->setIcon(icon);
    button->setIconSize(QSize(42, 42));

    connect(button, SIGNAL (pressed()), this, SLOT (on_btn_gaussian_blur_pressed()));

    QHBoxLayout *layout = new QHBoxLayout;
    layout->addWidget(button);

    ui->kernels->addLayout(layout);

    //set the layout of the kernel view
    ui->kernels->setAlignment(Qt::AlignTop);
    ui->kernels->setSpacing(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_pressed()
{
    //ui->setupUi(this);

    img = stbi_load( "/Users/Pieter/zebra_2.jpg", &w, &h, &comp, STBI_rgb );

    cl_uint platform_id_count = 0;
    clGetPlatformIDs( 0, nullptr, &platform_id_count );

    if(img == NULL)
    {
        QTextStream(stdout) << "Couldn't load image\n";
    }

    QTextStream(stdout) << "Loaded img: " << w << ", " << h << ", " << comp << '\n';

    QImage imageQ(img, w, h, QImage::Format_RGB888);
    //image.load("/Users/Pieter/Desktop/Schermafbeelding 2018-02-10 om 13.14.20.png");
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    QLabel* title = new QLabel("Grayscale");
    title->setStyleSheet("border-bottom: 5px solid #f5f6fa; text-transform: uppercase; color: #f5f6fa");

    ui->kernels->addWidget(title);
    //window.setLayout(layout);

    scene->setSceneRect(image.rect());

    QTimer::singleShot(200, this, SLOT(resize()));
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
    for( i = 0; i < w * h * comp; i += 3)
    {
        img[i + 1] = img[i];
        img[i + 2] = img[i];
    }

    QImage imageQ(img, w, h, QImage::Format_RGB888);
    //image.load("/Users/Pieter/Desktop/Schermafbeelding 2018-02-10 om 13.14.20.png");
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());
}

void MainWindow::on_btn_gaussian_blur_pressed()
{
    unsigned char* img_original( new unsigned char[ w * h * comp]);
    memcpy( img_original, img, w*h*3*sizeof(unsigned char) );

    int **gKernel;
    gKernel = new int *[kernel_size];
    for(int i = 0; i <kernel_size; i++)
        gKernel[i] = new int[kernel_size];
    int sum = createFilter(gKernel, kernel_size);

    for (int x = 0; x < kernel_size; ++x)
    {
        for (int y = 0; y < kernel_size; ++y)
        {
            QTextStream(stdout) << gKernel[x][y] << " ";
        }
        QTextStream(stdout) << "\n";

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

    QImage imageQ(img, w, h, QImage::Format_RGB888);
    scene = new QGraphicsScene(this);
    scene->addPixmap(QPixmap::fromImage(imageQ));
    ui->mainImage->setScene(scene);

    scene->setSceneRect(image.rect());

    delete [ ] img_original;
    img_original = NULL;

    for(int i = 0; i < kernel_size; i++)
        delete [ ] gKernel[i];
    delete [ ] gKernel;
    gKernel = NULL;

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
    for( i = 0; i < w * h * comp; i += 3)
    {
        img[i + 0] = std::min( 255, std::max( 0, (int) ( 255.0 * img[i + 0] / r ) ) );
        img[i + 1] = std::min( 255, std::max( 0, (int) (  255.0 * img[i + 1] / g ) ) );
        img[i + 2] = std::min( 255, std::max( 0, (int) (  255.0 * img[i + 2] / b ) ) );
    }

    QImage imageQ(img, w, h, QImage::Format_RGB888);
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
