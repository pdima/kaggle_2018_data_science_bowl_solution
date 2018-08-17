#ifndef SELECTIONCROPVIEW_H
#define SELECTIONCROPVIEW_H

#include <QWidget>

#include "selectionmodel.h"

class SelectionCropView : public QWidget
{
    Q_OBJECT
public:
    explicit SelectionCropView(QWidget *parent = 0);

    void setModel(SelectionModel* model);

    // QWidget interface
protected:
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;
    void paintEvent(QPaintEvent *) override;

signals:

public slots:
    void updateCrop();

private:
    double imgScale() const;
    QPointF configToWidget(const QPointF &p) const;
    QPointF widgetToConfig(const QPointF &p) const;

    SelectionModel* m_model;

    QImage m_crop;

    QPointF m_headPoint;
    QPointF m_tailPoint;
    bool m_drawing;
};

#endif // SELECTIONCROPVIEW_H
