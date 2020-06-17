import logging
import os
import sys

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QTextStream, Qt, QTimer, pyqtSignal
# make the example runnable without the need to install
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, qApp, QMessageBox, QAbstractItemView, QMainWindow, QApplication
from BreezeStyleSheets import breeze_resources

from test import arg_from_ui
import gui


class Steam:
    def __init__(self, view):
        self.view = view

    def write(self, *args):
        self.view.append(*args)

    def flush(self):
        return


class Signals(object):
    @staticmethod
    def open_pic(ui, text):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        try:
            if text == '打开图片':
                fileName, _ = QFileDialog.getOpenFileName(None, "打开图片", "",
                                                          "JPEG图片 (*.jpg);;PNG图片(*.png)", options=options)
            elif text == '打开图片文件夹':
                fileName = QFileDialog.getExistingDirectory(None, "打开图片文件夹", "", options=options)
            if fileName:
                print('选择路径：', fileName)
                ui.lineEdit.setText(fileName)
        except ValueError:
            print('没有选择文件')

    @staticmethod
    def stack_images(directory, layer_set):
        if len(layer_set) == 0:
            return None
        for index, item in enumerate(layer_set):
            if index == 0:
                back = Image.open(os.path.join(directory, item))
                continue
            fore = Image.open(os.path.join(directory, item))
            back = Image.alpha_composite(back, fore)
        return ImageQt(back)

    def save_segmentation(self, seg_dir, layer_set):
        if hasattr(self, 'seg_dir'):
            for index, item in enumerate(layer_set):
                if index == 0:
                    back = Image.open(os.path.join(seg_dir, item))
                    continue
                fore = Image.open(os.path.join(seg_dir, item))
                back = Image.alpha_composite(back, fore)
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
            try:
                fileName, _ = QFileDialog.getSaveFileName(None, "保存当前显示分割结果", "",
                                                          "PNG图片(*.png)", options=options)
                if fileName:
                    if fileName.find('.png') == -1:
                        back.save(fileName + '.png')
                        print('保存至', fileName + '.png')
                    else:
                        back.save(fileName)
                        print('保存至', fileName)
            except ValueError:
                print('保存时出现未知错误')

    @staticmethod
    def show_author():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("作者：Mr.ET")
        msg.setInformativeText("作者邮箱：1900432020@email.szu.edu.cn")
        msg.setWindowTitle("作者信息")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    gpu_mode = None
    def show_gpu_information(self, mode):
        if self.gpu_mode != mode:
            self.gpu_mode = mode
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            if mode == 'CPU':
                msg.setText("CPU处理一幅图像较慢，视硬件不同大概需要20秒到1分钟")
                msg.setInformativeText("由于处理较慢，执行过程中程序可能出现鼠标转圈的假死现象，请耐心等待")
                msg.setWindowTitle("使用CPU进行分割")
            elif mode == 'GPU':
                msg.setText("使用GPU分割需安装CUDA环境（本程序使用CUDA10.1环境进行开发），否则仍然会用CPU进行处理，安装链接：https://developer.nvidia.com/cuda-downloads")
                msg.setInformativeText("显存小于8GB的显卡处理某些图片可能会爆显存，这种情况下建议使用CPU处理该图片")
                msg.setWindowTitle("使用GPU进行分割")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def change_seg_folder(self, ui):
        self.layer_set.clear()
        for file in os.listdir(self.seg_dir):
            if file.endswith(".png"):
                if file.find('seg') != -1 or file.find('org') != -1:
                    continue
                self.layer_set.add(file)

        ui.listWidget.clear()
        ui.listWidget.addItems(list(self.layer_set))
        self.refresh(ui)

    def change_seg_combo_box(self, ui):
        self.seg_dir = os.path.join(self.result, ui.seg_combo_box.currentText())
        self.change_seg_folder(ui)

    def refresh(self, ui, change=True):
        if hasattr(self, 'seg_dir'):
            if change:
                self.qim = self.stack_images(self.seg_dir, self.layer_set)
            if self.qim:
                pix_map = QPixmap.fromImage(self.qim)
                scaled_pix_map = pix_map.scaled(ui.show_layers.size(), Qt.KeepAspectRatio)
                ui.show_layers.setPixmap(scaled_pix_map)
            else:
                # 没有图层显示时直接清空
                ui.show_layers.clear()

    def change_layers(self, ui, operation):
        if len(ui.listWidget.selectedItems()) == 0:
            return
        for item in ui.listWidget.selectedItems():
            # print(item.text())
            item_name = item.text()
            # 隐藏操作
            if operation == 'hide':
                if item_name in self.layer_set:
                    ui.listWidget.findItems(item_name, Qt.MatchExactly)[0].setForeground(Qt.gray)
                    self.layer_set.remove(item_name)
            # 显示操作
            elif operation == 'show':
                if item_name not in self.layer_set:
                    self.layer_set.add(item_name)
                    if ui.mode == 'dark':
                        ui.listWidget.findItems(item_name, Qt.MatchExactly)[0].setForeground(Qt.white)
                    elif ui.mode == 'light':
                        ui.listWidget.findItems(item_name, Qt.MatchExactly)[0].setForeground(Qt.black)
        self.refresh(ui)

    def seg_confirm(self, ui):
        if ui.gpu_combo_box.currentText() == 'CPU':
            gpu_flag = False
            print('CPU处理一幅图像较慢，视硬件不同大概需要20秒到1分钟，请耐心等待')
        elif ui.gpu_combo_box.currentText() == 'GPU':
            gpu_flag = True
            print('使用GPU需安装CUDA环境否则仍然会用CPU进行处理，且显存小于8GB的显卡处理某些图片可能会爆显存，爆显存的图片建议使用CPU处理该图片')
        self.show_gpu_information(ui.gpu_combo_box.currentText())
        imgs = ui.lineEdit.text()
        if not os.path.exists(imgs):
            print('没有找到' + str(ui.lineEdit.text()) + '图片或图片文件夹，请检查路径是否正确')
            return
        model = ui.model_combo_box.currentText()
        if not os.path.exists(model):
            print('在程序目录下没有找到' + str(model) + '文件夹，请检查是否已经下载选择的模型并放入程序目录下')
            return
        cfg_path = os.path.join('config', str(model) + '.yaml')
        if not os.path.exists(cfg_path):
            print('没有找到' + str(cfg_path) + '配置文件，请检查是否已经下载yaml配置文件并放入config目录下')
            return
        pth_dict = {'ade20k-hrnetv2': 30, 'ade20k-mobilenetv2dilated-c1_deepsup': 20,
                    'ade20k-resnet18dilated-c1_deepsup': 20,
                    'ade20k-resnet18dilated-ppm_deepsup': 20, 'ade20k-resnet50dilated-ppm_deepsup': 20,
                    'ade20k-resnet50-upernet': 30, 'ade20k-resnet101dilated-ppm_deepsup': 25,
                    'ade20k-resnet101-upernet': 50}
        encoder_path = os.path.join(model, 'encoder_epoch_' + str(pth_dict[model]) + '.pth')
        decoder_path = os.path.join(model, 'decoder_epoch_' + str(pth_dict[model]) + '.pth')
        if not os.path.exists(encoder_path):
            print('没有找到' + str(encoder_path) + 'pth文件，请检查是否已经下载选择的模型并放入程序目录下')
            return
        if not os.path.exists(decoder_path):
            print('没有找到' + str(decoder_path) + 'pth文件，请检查是否已经下载选择的模型并放入程序目录下')
            return

        for file in os.listdir(model):
            if file.endswith(".pth"):
                checkpoint = file[file.find('epoch'):]
                break
        self.result = 'segmentation'
        ui.progressBar.setValue(0)
        arg_from_ui(imgs=imgs, progress=ui.progressBar, gpu_flag=gpu_flag, config_path=cfg_path,
                    dir=model, checkpoint=checkpoint, result=self.result)

        self.layer_set = set()
        # 如果是文件夹，需要将图片添加到选择框中
        if os.path.isdir(imgs):
            ui.seg_combo_box.clear()
            for file in os.listdir(imgs):
                if file.endswith(".png") or file.endswith(".jpg"):
                    ui.seg_combo_box.addItem(os.path.splitext(file)[0])
            self.seg_dir = os.path.join(self.result, os.path.splitext(os.path.basename(os.listdir(imgs)[0]))[0])

        # 假如输入单张图片直接显示
        elif os.path.isfile(imgs):
            self.seg_dir = os.path.join(self.result, os.path.splitext(os.path.basename(imgs))[0])
            self.change_seg_folder(ui)
        ui.display_button.setEnabled(True)
        ui.hide_button.setEnabled(True)
        ui.save_seg_button.setEnabled(True)

    def register_signal(self, app, window, ui):
        sys.stdout = Steam(ui.textBrowser)
        ui.browser_button.clicked.connect(lambda: self.open_pic(ui, ui.open_pic_combo_box.currentText()))
        ui.open_pic.triggered.connect(lambda: self.open_pic(ui, '打开图片'))
        ui.open_pic_folder.triggered.connect(lambda: self.open_pic(ui, '打开图片文件夹'))
        ui.seg_confirm_button.clicked.connect(lambda: self.seg_confirm(ui))
        ui.output_empty_button.clicked.connect(lambda: ui.textBrowser.setText(''))
        ui.show_layers.setMinimumSize(1, 1)
        ui.show_layers.setAlignment(Qt.AlignCenter)
        ui.listWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        ui.display_button.clicked.connect(lambda: self.change_layers(ui, 'show'))
        ui.hide_button.clicked.connect(lambda: self.change_layers(ui, 'hide'))
        ui.save_seg_button.clicked.connect(lambda x: self.save_segmentation(self.seg_dir, self.layer_set))
        ui.save_seg.triggered.connect(
            lambda x: self.save_segmentation(self.seg_dir, self.layer_set) if hasattr(self, 'seg_dir') else x)
        window.resized.connect(lambda: self.refresh(ui, False))
        ui.seg_combo_box.currentTextChanged.connect(lambda: self.change_seg_combo_box(ui))
        ui.action_dark.triggered.connect(lambda: set_theme(app, ui, 'dark'))
        ui.action_light.triggered.connect(lambda: set_theme(app, ui, 'light'))
        ui.author.triggered.connect(lambda: self.show_author())
        ui.exit.triggered.connect(qApp.quit)


class Window(QMainWindow):
    resized = pyqtSignal()

    def __init__(self, parent=None):
        super(Window, self).__init__(parent=parent)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(Window, self).resizeEvent(event)


def set_theme(app, ui, mode):
    ui.mode = mode
    if mode in ['dark', 'light']:
        file = QFile(":/" + mode + ".qss")
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())


def main():
    """
    Application entry point
    """
    # logging.basicConfig(level=logging.DEBUG)
    # create the application and the main window
    app = QApplication(sys.argv)
    # app.setStyle(QtWidgets.QStyleFactory.create("fusion"))
    # window = QtWidgets.QMainWindow()
    window = Window()

    # 设置UI界面
    ui = gui.Ui_MainWindow()
    ui.setupUi(window)
    # 设置默认主题为dark
    set_theme(app, ui, 'dark')

    # register multiple signals
    signals = Signals()
    signals.register_signal(app, window, ui)

    # auto quit after 2s when testing on travis-ci
    # if "--travis" in sys.argv:
    #     QTimer.singleShot(2000, app.exit)

    # run
    window.show()
    # app.exec_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
