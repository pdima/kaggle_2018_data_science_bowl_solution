#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

#include <QObject>
#include <QFileInfo>
#include <QImage>
#include <QRectF>
#include <QBitmap>
#include <QUuid>

struct SelectionInfo
{
    QRect r;
    QImage mask;
    QImage outline;
    QString uuid;
    bool isDirty;

    void updateOutline(const QColor& outlineColor, int scale=1);
};

class SelectionModel: public QObject
{
    Q_OBJECT
public:
    SelectionModel();
    ~SelectionModel();

    static QString imagePath(const QFileInfo& imgDir);
    static QString hintImagePath(const QFileInfo& imgDir);
    static QString maskDir(const QFileInfo& imgDir);

    QFileInfo m_imgDir;
    QImage m_fullImage;
    QImage m_hintImage;
    QList<SelectionInfo> m_selections;
    QColor m_outlineColor {Qt::white};
    int m_currentSelection {-1};

    bool isEmpty() const { return m_selections.empty(); }
    int size() const { return m_selections.size(); }
    int currentSelectionIdx() const { return m_currentSelection; }

    SelectionInfo currentSelection() const;
    SelectionInfo& currentSelection() { return m_selections[m_currentSelection]; }

    void selectNextCrop();
    void selectPrevCrop();

    void setOutlineColor(const QColor& c);

    void appendSelection(const QRectF& r);

    void load(const QFileInfo& imgDir);
    void save();

    void exportCurrentCropAs(const QFileInfo& destDir);

public slots:
    void update();
    void setCurrentSelection(int selection);

    void clearSelections();
    void clearCurrentSelection();

signals:
    void loaded();
    void changed();
    void nextImageRequested();
    void prevImageRequested();

private:
    int boundSelIndex(int idx) const;
};

#endif // SELECTIONMODEL_H
