#include "selectiondetailsview.h"
#include "ui_selectiondetailsview.h"

#include "selectionmodel.h"

#include <QDebug>

SelectionDetailsView::SelectionDetailsView(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SelectionDetailsView)
{
    ui->setupUi(this);

//    connect(ui->checkSmallPart, SIGNAL(clicked(bool)), SLOT(updateModelFromBoxes()));
//    connect(ui->checkWrongSpecies, SIGNAL(clicked(bool)), SLOT(updateModelFromBoxes()));
//    connect(ui->checkVeryLowQuality, SIGNAL(clicked(bool)), SLOT(updateModelFromBoxes()));
//    connect(ui->checkIgnored, SIGNAL(clicked(bool)), SLOT(updateModelFromBoxes()));
//    connect(ui->checkUnsure, SIGNAL(clicked(bool)), SLOT(updateModelFromBoxes()));

//    m_species.addButton(ui->radioButton_0, 0);
//    m_species.addButton(ui->radioButton_A, 1);
//    m_species.addButton(ui->radioButton_B, 2);
//    m_species.addButton(ui->radioButton_D, 3);
//    m_species.addButton(ui->radioButton_L, 4);
//    m_species.addButton(ui->radioButton_O, 5);
//    m_species.addButton(ui->radioButton_S, 6);
//    m_species.addButton(ui->radioButton_Y, 7);
    m_species.setExclusive(true);

    m_speciesNames = QStringList({"", "ALB", "BET", "DOL", "LAG", "OTHER", "SHARK", "YFT"});
    connect(&m_species, SIGNAL(buttonClicked(int)), SLOT(updateModelFromBoxes()));
}

SelectionDetailsView::~SelectionDetailsView()
{
    delete ui;
}

void SelectionDetailsView::setModel(SelectionModel *model)
{
    m_model = model;
    ui->selectionMaskView->setModel(model);

    connect(model, SIGNAL(changed()), SLOT(updateBoxes()));
}

void SelectionDetailsView::updateBoxes()
{
    if (m_model->isEmpty())
    {
        setEnabled(false);
//        ui->checkSmallPart->setChecked(false);
//        ui->checkWrongSpecies->setChecked(false);
//        ui->checkVeryLowQuality->setChecked(false);
//        ui->checkIgnored->setChecked(false);
//        ui->checkUnsure->setChecked(false);
    }
    else
    {
        setEnabled(true);
//        ui->checkSmallPart->setChecked(m_model->currentSelection().smallPart);
//        ui->checkWrongSpecies->setChecked(m_model->currentSelection().wrongSpecies);
//        ui->checkVeryLowQuality->setChecked(m_model->currentSelection().lowQuality);
//        ui->checkIgnored->setChecked(m_model->currentSelection().ignored);
//        ui->checkUnsure->setChecked(m_model->currentSelection().unsure);

//        int speciesId = m_speciesNames.indexOf(m_model->currentSelection().species);
//        m_species.button(speciesId)->setChecked(true);
//        ui->speciesComboBox->setCurrentText(m_model->currentSelection().species);
//        qDebug() << "Species:" << m_model->currentSelection().species;
    }
}

void SelectionDetailsView::updateModelFromBoxes()
{
//    m_model->currentSelection().smallPart = ui->checkSmallPart->isChecked();
//    m_model->currentSelection().wrongSpecies = ui->checkWrongSpecies->isChecked();
//    m_model->currentSelection().lowQuality = ui->checkVeryLowQuality->isChecked();
//    m_model->currentSelection().ignored = ui->checkIgnored->isChecked();
//    m_model->currentSelection().unsure = ui->checkUnsure->isChecked();
//    m_model->currentSelection().species = m_speciesNames[m_species.checkedId()];

    m_model->save();
    m_model->update();
}
