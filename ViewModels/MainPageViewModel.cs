using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ILGPU;
using ILGPU.Runtime;
using NeuralNetwork;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DigitRecognition.ViewModels {
    internal partial class MainPageViewModel : ObservableObject {

        private static readonly int[] layers = new int[] { 784, 512, 128, 32, 10 };

        private Context gpuContext;
        public Context GpuContext => gpuContext ??=
            Context.Create(i => i.Default().EnableAlgorithms());

        [ObservableProperty]
        ObservableCollection<Device> devices;

        private Accelerator accelerator;
        private Brain brain;

        [RelayCommand]
        private void InitDevices() {
            Devices = new ObservableCollection<Device>(GpuContext.Devices);

            SelectedDevice = string.IsNullOrEmpty(SavedSelectedDevice)
                ? Devices.FirstOrDefault()
                : Devices.FirstOrDefault(d => d.Name == SavedSelectedDevice);
        }

        public string SavedSelectedDevice {
            get {
                return Properties.Settings.Default.SelectedDevice;
            }
            set {
                Properties.Settings.Default.SelectedDevice = value;
                Properties.Settings.Default.Save();
            }
        }

        private Device selectedDevice;
        public Device SelectedDevice {
            get => selectedDevice; set {
                selectedDevice = value;
                OnPropertyChanged();
                if (selectedDevice == null)
                    return;

                SavedSelectedDevice = selectedDevice.Name;
                accelerator?.Dispose();
                accelerator = selectedDevice.CreateAccelerator(GpuContext);
                brain = new Brain(accelerator, 1e-3f, layers);
            }
        }

        [RelayCommand]
        private void Save() {
            brain.Save(Properties.Settings.Default.PathToSavings + @"\nn.bin");
            MessageBox.Show("SAVED!");
        }

        [RelayCommand]
        private void Read() {
            brain.Read(Properties.Settings.Default.PathToSavings + @"\nn.bin");
            MessageBox.Show("READ!");
        }

        [RelayCommand]
        private void Clear(object obj) {

            var inkCanvas = (InkCanvas)obj;
            inkCanvas.Strokes.Clear();
        }

        public static event Action OpeningSettings;

        [RelayCommand]
        private void OpenSettings() {
            OpeningSettings?.Invoke();
        }

        private record BrainSettings(string PathToDataset, int EpochsQuantity, int BatchSize) : IBrainSettings;

        [RelayCommand]
        private async Task RunLearning() {

            if (brain == null)
                return;

            var settings = Properties.Settings.Default;
            await Task.Run(() => {

                var brainSettings = new BrainSettings(settings.PathToDataset, settings.EpochsQuantity, settings.BatchSize);
                brain.Train(brainSettings, accelerator);
            });

            MessageBox.Show("DONE!!!");
        }

        [RelayCommand]
        private void Recognize(object obj) {

            if (brain == null)
                return;

            var inkCanvas = (InkCanvas)obj;
            var target = new RenderTargetBitmap((int)inkCanvas.Width, (int)inkCanvas.Height, 96d, 96d, PixelFormats.Default);

            target.Render(inkCanvas);
            int pixelSize = target.Format.BitsPerPixel / 8;
            int stride = target.PixelWidth * pixelSize;
            int length = stride * target.PixelHeight;
            var imgBin = new byte[length];
            target.CopyPixels(imgBin, stride, 0);

            int result = brain.Recognize(imgBin, pixelSize, accelerator);
            MessageBox.Show($"Я думаю, что это {result}");
        }
    }
}
