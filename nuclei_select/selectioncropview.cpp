#include "selectioncropview.h"
#include "selectionmodel.h"

#include <QMouseEvent>
#include <QPainter>

SelectionCropView::SelectionCropView(QWidget *parent) : QWidget(parent)
{
    setMouseTracking(true);
}

void SelectionCropView::setModel(SelectionModel *model)
{
    m_model = model;
    connect(model, SIGNAL(changed()), SLOT(updateCrop()));
}

void SelectionCropView::mousePressEvent(QMouseEvent *ev)
{
    QWidget::mousePressEvent(ev);

    if (ev->button() == Qt::LeftButton)
    {
        if (m_drawing)
        {
            m_drawing = false;
//            m_model->currentSelection().head = m_headPoint;
//            m_model->currentSelection().tail = widgetToConfig(ev->pos());

            m_model->save();
            m_model->update();
        }
        else
        {
            m_drawing = true;
            m_headPoint = widgetToConfig(ev->pos());
        }
    }
}

void SelectionCropView::mouseMoveEvent(QMouseEvent *ev)
{
    QWidget::mouseMoveEvent(ev);
    if (m_drawing)
    {
        m_tailPoint = widgetToConfig(ev->pos());
        update();
    }
}


void SelectionCropView::wheelEvent(QWheelEvent *ev)
{
    if (ev->angleDelta().y() < 0)
        m_model->selectNextCrop();

    if (ev->angleDelta().y() > 0)
        m_model->selectPrevCrop();
}

void SelectionCropView::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    if (!m_crop.isNull())
    {
        double scale = imgScale();

        QRectF scaledRect = QRectF(0, 0, m_crop.width()*scale, m_crop.height()*scale);
        p.drawImage(scaledRect, m_crop);
    }

    QPointF head;
    QPointF tail;

    if (m_drawing)
    {
        head = m_headPoint;
        tail = m_tailPoint;
    }
    else
    {
        if (!m_model->isEmpty())
        {
            head = m_model->currentSelection().head;
            tail = m_model->currentSelection().tail;
        }
    }

    if (head != QPointF() && tail != QPointF())
    {
        QPen pen = p.pen();
        pen.setColor(Qt::white);
        pen.setWidthF(3.0);

        p.setPen(pen);
        p.drawLine(configToWidget(head), configToWidget(tail));

        pen.setColor(Qt::darkGray);
        pen.setWidthF(4.0);
        p.setPen(pen);
        p.drawEllipse(configToWidget(head), 3, 3);

        pen.setColor(Qt::white);
        pen.setWidthF(1.0);
        p.setPen(pen);
        p.drawEllipse(configToWidget(head), 4, 4);
    }
}

void SelectionCropView::updateCrop()
{
    if (m_model->isEmpty())
    {
        m_crop = QImage();
    }
    else
    {
        m_crop = m_model->m_fullImage.copy(m_model->currentSelection().r.toRect());
    }

    update();
}


double SelectionCropView::imgScale() const
{
    if (m_crop.isNull())
        return 1.0;

    return qMin(double(width())/m_crop.width(), double(height())/m_crop.height());
}

QPointF SelectionCropView::configToWidget(const QPointF &p) const
{
    double s = imgScale();
    return QPointF(p.x()*s, p.y()*s);
}

QPointF SelectionCropView::widgetToConfig(const QPointF &p) const
{
    double s = imgScale();
    return QPointF(p.x()/s, p.y()/s);
}
