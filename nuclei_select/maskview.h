#ifndef MASKVIEW_H
#define MASKVIEW_H

#include <QWidget>
#include <QBitmap>

#include "selectionmodel.h"

class MaskView : public QWidget
{
    Q_OBJECT
public:
    explicit MaskView(QWidget *parent = 0);

    void setModel(SelectionModel* model);

signals:

public slots:
    void updateCrop();
    void setOutlineDisplayed(bool displayed);
    void setHintsDisplayed(bool displayed);

    // QWidget interface
protected:
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;
    void paintEvent(QPaintEvent *) override;

private:
    double imgScale() const;
    QPointF configToWidget(const QPointF &p) const;
    QPointF widgetToConfig(const QPointF &p) const;

    void paintOnMask(QMouseEvent* ev, int size);

    SelectionModel* m_model;

    SelectionInfo m_selection;
    QImage m_crop;

    QPoint m_mousePos;
    double m_brushSize {64};
    int m_border{64};
    bool m_displayOutlines {true};
    bool m_displayHints {true};
};

#endif // MASKVIEW_H
