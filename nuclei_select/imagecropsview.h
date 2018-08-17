#ifndef IMAGECROPSVIEW_H
#define IMAGECROPSVIEW_H


#include <QWidget>
#include <QFileInfo>

class SelectionModel;

class ImageCropsView : public QWidget
{
    Q_OBJECT
public:
    ImageCropsView(QWidget *parent);
    virtual ~ImageCropsView();

    void setModel(SelectionModel* model);
    void setReferenceModel(SelectionModel* model);

    // QWidget interface
protected:
    void wheelEvent(QWheelEvent *) override;
    void paintEvent(QPaintEvent *) override;

private:
    double imgScale() const;

    SelectionModel* m_model {nullptr};
    SelectionModel* m_backgroundModel {nullptr};

    bool m_fixedScale {false};
    bool m_selecting;
    QPointF m_startSelection;
    QPointF m_endSelection;
};


#endif // IMAGECROPSVIEW_H
