#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <vector>

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

    std::vector<int*> enabled_devices;

private slots:
    void on_pushButton_pressed();
    void on_horizontalSlider_sliderMoved(int position);
    void on_pushButton_2_pressed();
    void on_pushButton_3_pressed();
    void on_horizontalSlider_valueChanged(int value);
    void on_pushButton_4_pressed();
    void on_horizontalSlider_4_valueChanged(int value);
    void on_horizontalSlider_3_valueChanged(int value);
    void on_horizontalSlider_2_valueChanged(int value);
};


#endif // MAINWINDOW_H
