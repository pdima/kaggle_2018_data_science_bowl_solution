#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QInputDialog>
#include <QSettings>
#include <QDebug>
#include <QColorDialog>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->maskView->setModel(&m_model);
    ui->fullImageView->setModel(&m_model);
    ui->splitter->setStretchFactor(0, 2);
    ui->splitter->setStretchFactor(1, 1);

    connect(ui->actionExit, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
    connect(ui->actionOpen_Directory, SIGNAL(triggered(bool)), SLOT(openDirectory()));
    connect(ui->actionNext_Image, SIGNAL(triggered(bool)), SLOT(nextImage()));
    connect(ui->actionPrev_Image, SIGNAL(triggered(bool)), SLOT(prevImage()));
    connect(ui->actionSelect_Image, SIGNAL(triggered(bool)), SLOT(selectImage()));
    connect(ui->actionDelete_selection, SIGNAL(triggered(bool)), &m_model, SLOT(clearCurrentSelection()));
    connect(ui->actionDisplay_outlines, SIGNAL(triggered(bool)), ui->fullImageView, SLOT(setOutlineDisplayed(bool)));
    connect(ui->actionDisplay_rects, SIGNAL(triggered(bool)), ui->fullImageView, SLOT(setRectDisplayed(bool)));
    connect(ui->actionDisplay_masks, SIGNAL(triggered(bool)), ui->fullImageView, SLOT(setMaskDisplayed(bool)));
    connect(ui->actionDisplay_hints, SIGNAL(triggered(bool)), ui->fullImageView, SLOT(setHintsDisplayed(bool)));

    connect(ui->actionDisplay_outlines, SIGNAL(triggered(bool)), ui->maskView, SLOT(setOutlineDisplayed(bool)));
    connect(ui->actionDisplay_hints, SIGNAL(triggered(bool)), ui->maskView, SLOT(setHintsDisplayed(bool)));

    connect(ui->actionSet_Mask_Color, SIGNAL(triggered(bool)), SLOT(setMaskCOlor()));
    connect(ui->actionExport_current_crop_as, SIGNAL(triggered(bool)), SLOT(exportCurrentCrop()));

//    connect(&m_model, SIGNAL(nextImageRequested()), SLOT(nextImage()));
//    connect(&m_model, SIGNAL(prevImageRequested()), SLOT(prevImage()));

    QSettings s;
    QColor outlineColor = s.value("mask_color", QColor(Qt::white)).value<QColor>();
    m_model.setOutlineColor(outlineColor);

//    openDirectory(lastDir());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openDirectory()
{
    openDirectory(QFileDialog::getExistingDirectory());
}

void MainWindow::openDirectory(const QString& d)
{
    m_dir = d;
    setLastDir(m_dir);

    QDir dir(m_dir);
    m_files = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    selectImage(0);
}

void MainWindow::selectImage()
{
    QStringList items;
    for (auto fi: m_files)
    {
        items.append(fi.fileName());
    }

    QString item = QInputDialog::getItem(this, "Select Image", "Image to jump to", items, m_currentFile);

    for (int i=0; i<items.size(); i++)
    {
        if (item == items[i])
            selectImage(i);
    }
}

void MainWindow::selectImage(int idx)
{
    if (idx >= 0 && idx < m_files.count())
    {
        m_currentFile = idx;
        m_model.load(m_files[m_currentFile]);
        setWindowTitle(QString("%1 %2  %3/%4").arg(m_files[m_currentFile].fileName()).arg(m_dir).arg(idx+1).arg(m_files.count()));
    }
}

void MainWindow::prevImage()
{
    selectImage(m_currentFile-1);
}

void MainWindow::nextImage()
{
    selectImage(m_currentFile+1);
}


QString MainWindow::lastDir() const
{
    QSettings s;
    return s.value("last_dir", ".").toString();
}

void MainWindow::setLastDir(const QString &dir)
{
    QSettings s;
    s.setValue("last_dir", dir);
}

void MainWindow::setMaskCOlor()
{
    QSettings s;
    QColor color = s.value("mask_color", QColor(Qt::white)).value<QColor>();

    color = QColorDialog::getColor(color);
    if (color.isValid())
    {
        s.setValue("mask_color", color);
        m_model.setOutlineColor(color);
    }
}

void MainWindow::exportCurrentCrop()
{
    QString cropName = QInputDialog::getText(this, "Crop name", "Crop name", QLineEdit::Normal,
                                             m_files[m_currentFile].fileName()+"_crop");

    if (!cropName.isEmpty())
    {
        m_model.exportCurrentCropAs(QFileInfo(QDir(m_files[m_currentFile].absolutePath()), cropName));
    }
}
