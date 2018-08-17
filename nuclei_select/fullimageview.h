#ifndef SELECTIONONIMAGE_H
#define SELECTIONONIMAGE_H

#include <QWidget>
#include <QFileInfo>

class SelectionModel;

class FullImageView : public QWidget
{
    Q_OBJECT
public:
    FullImageView(QWidget *parent=0);
    virtual ~FullImageView();

    void setModel(SelectionModel* model);

signals:
    void nextImageRequested();
    void prevImageRequested();

public slots:
    void updateImage();
    void setOutlineDisplayed(bool displayed);
    void setRectDisplayed(bool displayed);
    void setMaskDisplayed(bool displayed);
    void setHintsDisplayed(bool displayed);

    // QWidget interface
protected:
    void mousePressEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;
    void paintEvent(QPaintEvent *) override;
    QSize sizeHint() const override;

private:
    double imgScale() const;
    QRectF imgToWidget(const QRectF& r);
    QRectF widgetToImg(const QRectF& r);
    QPointF widgetToImg(const QPointF& p);

    SelectionModel* m_model {nullptr};

    bool m_selecting {false};
    bool m_displayOutlines {true};
    bool m_displayRects {true};
    bool m_displayMasks {false};
    bool m_displayHints {true};
    double m_scale {2.0};
    QPointF m_startSelection;
    QPointF m_endSelection;
};

#endif // SELECTIONONIMAGE_H
