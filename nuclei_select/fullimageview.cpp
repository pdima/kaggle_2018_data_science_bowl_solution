#include "fullimageview.h"
#include "selectionmodel.h"

#include <QPainter>
#include <QMouseEvent>
#include <QTextStream>
#include <QDebug>

FullImageView::FullImageView(QWidget *parent)
    : QWidget(parent)
{
}

FullImageView::~FullImageView()
{
}

void FullImageView::setModel(SelectionModel *model)
{
    m_model = model;
    connect(model, SIGNAL(changed()), SLOT(update()));
    connect(model, SIGNAL(loaded()), SLOT(updateImage()));
}

void FullImageView::updateImage()
{
    resize(sizeHint());
    update();
}

void FullImageView::setOutlineDisplayed(bool displayed)
{
    m_displayOutlines = displayed;
    update();
}

void FullImageView::setRectDisplayed(bool displayed)
{
    m_displayRects = displayed;
    update();
}

void FullImageView::setMaskDisplayed(bool displayed)
{
    m_displayMasks = displayed;
    update();
}

void FullImageView::setHintsDisplayed(bool displayed)
{
    m_displayHints = displayed;
    update();
}

void FullImageView::mousePressEvent(QMouseEvent *ev)
{
    if (ev->button() == Qt::LeftButton)
    {
        m_selecting = true;
        m_startSelection = ev->pos();
        m_endSelection = ev->pos();
    }
}

void FullImageView::mouseReleaseEvent(QMouseEvent *ev)
{
    if (ev->button() == Qt::LeftButton)
    {
        m_endSelection = ev->pos();
        m_selecting = false;
        QRectF imgSelection = widgetToImg(QRectF(m_startSelection, m_endSelection));
        if (imgSelection.width() > 10 && imgSelection.height() > 10)
        {
            m_model->appendSelection(imgSelection);
        }
        else
        {
            for (int i=0; i<m_model->m_selections.size(); i++)
            {
                if (m_model->m_selections[i].r.contains(widgetToImg(m_endSelection).toPoint()))
                {
                    m_model->setCurrentSelection(i);
                    break;
                }
            }
        }
    }
}

void FullImageView::mouseMoveEvent(QMouseEvent *ev)
{
    if (m_selecting)
    {
        m_endSelection = ev->pos();
        update();
    }
}

void FullImageView::wheelEvent(QWheelEvent *ev)
{
    if (ev->angleDelta().y() < 0)
        m_scale *= 1.25;
//        emit nextImageRequested();

    if (ev->angleDelta().y() > 0)
        m_scale /= 1.25;
//        emit prevImageRequested();
    updateImage();
}

void FullImageView::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    QColor normalSelectionColor(4, 4, 100);
    QColor currentSelectionColor(255, 50, 50);

    if (!m_model->m_fullImage.isNull())
    {
        double scale = imgScale();

        QRectF scaledRect = QRectF(0, 0, m_model->m_fullImage.width()*scale, m_model->m_fullImage.height()*scale);

        if (m_displayHints)
            p.drawImage(scaledRect, m_model->m_hintImage);
        else
            p.drawImage(scaledRect, m_model->m_fullImage);

        for (int i=0; i<m_model->size(); i++)
        {
            if (i == m_model->currentSelectionIdx())
                p.setPen(currentSelectionColor);
            else
                p.setPen(normalSelectionColor);

            QRectF selectionInWidgetSpace = imgToWidget(m_model->m_selections[i].r);
//            qDebug() << selectionInWidgetSpace;

//            p.fillRect(selectionInWidgetSpace, QColor(40, 40, 100, 50));
            if (m_displayRects)
                p.drawRect(selectionInWidgetSpace);

            if (m_displayOutlines)
                p.drawImage(selectionInWidgetSpace, m_model->m_selections[i].outline);

            if (m_displayMasks)
            {
                p.setClipRegion(QRegion(QBitmap::fromImage(m_model->m_selections[i].mask.scaled(scaledRect.size().toSize()))));
//                p.fillRect(scaledRect, QBrush(QColor(0,250,0,128)));
                p.fillRect(scaledRect, QColor(0,250,0,256));
            }
//                p.drawImage(selectionInWidgetSpace, m_model->m_selections[i].mask);
        }

        if (m_selecting)
        {
            p.setPen(normalSelectionColor);
            p.fillRect(QRectF(m_startSelection, m_endSelection), QColor(40, 40, 100, 50));
            p.drawRect(QRectF(m_startSelection, m_endSelection));
        }
    }
}

QSize FullImageView::sizeHint() const
{
    if (m_model && !m_model->m_fullImage.isNull())
    {
        return m_model->m_fullImage.size()*m_scale;
    }

    return QSize(512, 512);
}

double FullImageView::imgScale() const
{
    return m_scale;

//    if (m_model->m_fullImage.isNull())
//        return 1.0;

//    return qMin(double(width())/m_model->m_fullImage.width(), double(height())/m_model->m_fullImage.height());
}

QRectF FullImageView::imgToWidget(const QRectF &r)
{
    double s = imgScale();
    return QRectF(r.topLeft()*s, r.bottomRight()*s).normalized();
}

QRectF FullImageView::widgetToImg(const QRectF &r)
{
    double s = imgScale();
    return QRectF(r.topLeft()/s, r.bottomRight()/s).normalized();
}

QPointF FullImageView::widgetToImg(const QPointF &p)
{
    double s = imgScale();
    return p/s;
}

