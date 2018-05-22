#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <vector>
#include <QPoint>
#include <QString>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void resizeEvent(QResizeEvent* event);
    void resize();
    void on_open_triggered();

private:
    void createFilter(double **gKernel, int size);
    int createFilter(int **gKernel, int size);
    int printDevices();
    void toggleDevice(int platform, int device, bool toggle);
    void execute_pipeline();
    void transfer_image();

    Ui::MainWindow *ui;
    QGraphicsScene *scene;
    QPixmap image;
    unsigned char *img = NULL;
    int w, h, comp;

    int r,g,b;

    int kernel_size = 5;

    bool enabled_gaussian = false;
    bool enabled_whitepoint = false;
    bool enabled_blackwhite = false;

    std::vector<int*> enabled_devices;

   QString temp_file;

    //QPoint dragStartPosition;

private slots:
    void on_horizontalSlider_sliderMoved(int position);
    void on_apply_pressed();
    void on_horizontalSlider_valueChanged(int value);
    void on_horizontalSlider_4_valueChanged(int value);
    void on_horizontalSlider_3_valueChanged(int value);
    void on_horizontalSlider_2_valueChanged(int value);
    void on_enable_blackwhite_stateChanged(int value);
    void on_enable_whitepoint_stateChanged(int value);
    void on_enable_gaussian_stateChanged(int value);
    //void mousePressEvent(QMouseEvent *event);
    //void mouseMoveEvent(QMouseEvent *event);
};


#endif // MAINWINDOW_H
