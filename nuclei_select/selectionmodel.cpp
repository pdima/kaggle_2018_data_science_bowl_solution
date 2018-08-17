#include "selectionmodel.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDir>
#include <QDebug>
#include <QPainter>
#include <QCryptographicHash>

SelectionModel::SelectionModel()
{
}

SelectionModel::~SelectionModel()
{

}

QString SelectionModel::imagePath(const QFileInfo &imgDir)
{
    QDir d(imgDir.absoluteFilePath());
    d.cd("images");
    return d.filePath(imgDir.baseName()+".png");
}

QString SelectionModel::hintImagePath(const QFileInfo &imgDir)
{
    QDir d(imgDir.absoluteFilePath());
    d.cd("images");
    return d.filePath("hint.png");
}

QString SelectionModel::maskDir(const QFileInfo &imgDir)
{
    QDir d(imgDir.absoluteFilePath());
    return d.filePath("masks");
}

SelectionInfo SelectionModel::currentSelection() const
{
    if (m_currentSelection >= 0 && m_currentSelection < m_selections.size())
        return m_selections[m_currentSelection];

    return SelectionInfo();
}

void SelectionModel::selectNextCrop()
{
    if (m_currentSelection < m_selections.size()-1)
    {
        setCurrentSelection(m_currentSelection+1);
    }
    else
    {
        emit nextImageRequested();
    }
}

void SelectionModel::selectPrevCrop()
{
    if (m_currentSelection > 0)
    {
        setCurrentSelection(m_currentSelection-1);
    }
    else
    {
        emit prevImageRequested();
    }
}

void SelectionModel::setOutlineColor(const QColor &c)
{
    m_outlineColor = c;
    for (SelectionInfo& selection: m_selections)
    {
        selection.updateOutline(c);
    }

    emit changed();
}

void SelectionModel::appendSelection(const QRectF &r)
{
    SelectionInfo info;
    info.r = r.toRect().intersected(QRect(0, 0, m_fullImage.width(), m_fullImage.height()));
    info.mask = QImage(info.r.size(), QImage::Format_Mono);
    info.mask.fill(Qt::color0);
    info.outline = QImage(info.r.size(), QImage::Format_ARGB32_Premultiplied);
    info.outline.fill(QColor(0, 0, 0, 0));
    info.isDirty = false;
    QCryptographicHash hash(QCryptographicHash::Sha1);
    hash.addData(QUuid::createUuid().toByteArray());
    info.uuid = QString::fromLatin1(hash.result().toHex());

//    info.mask.save("/tmp/mask_new.png");
//    info.outline.save("/tmp/outline_new.png");

    m_selections.append(info);
    m_currentSelection = m_selections.size() - 1;
    save();
    emit changed();
}


void SelectionModel::load(const QFileInfo &imgDir)
{
    m_imgDir = imgDir;
    m_fullImage = QImage(imagePath(imgDir));
    m_hintImage = QImage(hintImagePath(imgDir));
    if (m_hintImage.isNull())
        m_hintImage = m_fullImage;

    qDebug() << imagePath(imgDir);
    QDir masksDir(maskDir(imgDir));

    m_currentSelection = -1;
    m_selections.clear();

    for (QFileInfo maskFile: masksDir.entryInfoList(QDir::Files))
    {
        SelectionInfo info;
        QImage mask(maskFile.absoluteFilePath());
        int minX = mask.width()-1;
        int maxX = 0;
        int minY = mask.height()-1;
        int maxY = 0;

        for (int x=0; x<mask.width(); x++)
        {
            for (int y=0; y<mask.height(); y++)
            {
                if (mask.pixelColor(x, y) != Qt::black)
                {
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                }
            }
        }

        minX = std::max(0, minX-4);
        minY = std::max(0, minY-4);
        maxX = std::min(mask.width()-1, maxX+4);
        maxY = std::min(mask.height()-1, maxY+4);
        info.r = QRect(minX, minY, maxX-minX, maxY-minY);
        info.mask = mask.copy(info.r);
        info.updateOutline(m_outlineColor);
        info.isDirty = false;
        info.uuid = maskFile.baseName();

        qDebug() << maskFile.absoluteFilePath() << info.r;

        m_selections.append(info);
    }

    if (!m_selections.empty())
        m_currentSelection = 0;

    emit loaded();
    update();
}

void SelectionModel::save()
{
    QDir masksDir(maskDir(m_imgDir));
    masksDir.mkpath(masksDir.absolutePath());

    QSet<QString> selectionIds;

    for (SelectionInfo& info: m_selections)
    {
        selectionIds.insert(info.uuid);

        if (info.isDirty)
        {
            QImage fullMask(m_fullImage.size(), QImage::Format_Grayscale8);
            fullMask.fill(Qt::black);
            {
                QPainter p(&fullMask);
                p.drawImage(info.r, info.mask);
            }
            fullMask.save(masksDir.absoluteFilePath(info.uuid+".png"));
            qDebug() << "save mask" << info.uuid;
        }
    }

    for (QFileInfo maskFile: masksDir.entryInfoList(QDir::Files))
    {
        if (!selectionIds.contains(maskFile.baseName()))
        {
            qDebug() << "delete mask" << maskFile.baseName();
            masksDir.remove(maskFile.fileName());
        }
    }
}

void SelectionModel::exportCurrentCropAs(const QFileInfo &destDir)
{
    QRect crop = m_selections[m_currentSelection].r;
    QImage imgCrop = m_fullImage.copy(crop);

    QDir imgDir(destDir.absoluteFilePath());
    imgDir.mkpath(imgDir.absolutePath()+"/images");
    qDebug() << "imgDir.absolutePath()" << imgDir.absolutePath();
    qDebug() << "imgDir.absolutePath() fn" << imgDir.absoluteFilePath(destDir.fileName()+".png");

    imgCrop.save(imgDir.absoluteFilePath("images/"+destDir.fileName()+".png"));


    QDir masksDir(maskDir(destDir));
    masksDir.mkpath(masksDir.absolutePath());

    for (SelectionInfo& info: m_selections)
    {
        QImage fullMask(m_fullImage.size(), QImage::Format_Grayscale8);
        fullMask.fill(Qt::black);
        {
            QPainter p(&fullMask);
            p.drawImage(info.r, info.mask);
        }
        QImage mask = fullMask.copy(crop);

        bool found = false;
        for (int x=0; x<mask.width() && !found; x++)
        {
            for (int y=0; y<mask.height() && !found; y++)
            {
                if (mask.pixelColor(x,y) != Qt::black)
                    found = true;
            }
        }

        if (found)
        {
            mask.save(masksDir.absoluteFilePath(info.uuid+".png"));
            qDebug() << "save mask" << info.uuid;
        }
    }
}

void SelectionModel::update()
{
    emit changed();
}

void SelectionModel::setCurrentSelection(int selection)
{
    if (boundSelIndex(selection) != m_currentSelection)
    {
        m_currentSelection = boundSelIndex(selection);
        emit changed();
    }
}

void SelectionModel::clearSelections()
{
    m_selections.clear();
    m_currentSelection = -1;
    save();
    emit changed();
}

void SelectionModel::clearCurrentSelection()
{
    if (m_currentSelection >= 0 && m_currentSelection < m_selections.size())
    {
        m_selections.removeAt(m_currentSelection);

        if (m_currentSelection >= m_selections.size())
            m_currentSelection--;

        save();
        emit changed();
    }
}

int SelectionModel::boundSelIndex(int idx) const
{
    if (isEmpty())
        return -1;

    if (idx < 0)
        return 0;

    if (idx > m_selections.size()-1)
        return m_selections.size()-1;

    return idx;
}


void SelectionInfo::updateOutline(const QColor& outlineColor, int scale)
{
    int w = mask.width();
    int h = mask.height();

    if (outline.size() != mask.size()*scale)
    {
        outline = QImage(mask.size()*scale, QImage::Format_ARGB32_Premultiplied);
    }

    int whitePixels = 0;
    int blackPixels = 0;
    int borderPixels = 0;

    for (int x=0; x<w*scale; x++)
    {
        for (int y=0; y<h*scale; y++)
        {
            auto imgColor = mask.pixelColor(x/scale, y/scale);
            if (imgColor != Qt::black)
            {
                whitePixels++;
                bool onBorder =
                        (x > 0 && mask.pixelColor((x-1)/scale, y/scale) == Qt::black) ||
                        (x < w*scale-1 && mask.pixelColor((x+1)/scale, y/scale) == Qt::black) ||
                        (y > 0 && mask.pixelColor(x/scale, (y-1)/scale) == Qt::black) ||
                        (y < h*scale-1 && mask.pixelColor(x/scale, (y+1)/scale) == Qt::black);

                outline.setPixelColor(x, y,onBorder ? outlineColor : QColor(0, 0, 0, 0));
                if (onBorder)
                    borderPixels++;
            }
            else
            {
                blackPixels++;
                outline.setPixelColor(x, y, QColor(0, 0, 0, 0));
            }
        }
    }

//    qDebug() << "wb" << whitePixels << blackPixels << borderPixels;
}
