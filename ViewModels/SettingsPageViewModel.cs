using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Windows.Forms;

namespace DigitRecognition.ViewModels {
    internal partial class SettingsPageViewModel : ObservableObject {

        public static event Action OnExit;

        [RelayCommand]
        private void Exit() {
            OnExit?.Invoke();
        }

        public int EpochsQuantity {
            get {
                return Properties.Settings.Default.EpochsQuantity;
            }
            set {
                Properties.Settings.Default.EpochsQuantity = value;
                Properties.Settings.Default.Save();
                OnPropertyChanged();
            }
        }

        public int BatchSize {
            get {
                return Properties.Settings.Default.BatchSize;
            }
            set {
                Properties.Settings.Default.BatchSize = value;
                Properties.Settings.Default.Save();
                OnPropertyChanged();
            }
        }

        public string PathToDataset {
            get {
                return Properties.Settings.Default.PathToDataset;
            }
            set {
                Properties.Settings.Default.PathToDataset = value;
                Properties.Settings.Default.Save();
                OnPropertyChanged();
            }
        }

        public string PathToSavings {
            get {
                return Properties.Settings.Default.PathToSavings;
            }
            set {
                Properties.Settings.Default.PathToSavings = value;
                Properties.Settings.Default.Save();
                OnPropertyChanged();
            }
        }

        [RelayCommand]
        private void ChangePathToDataset() {
            var browser = new FolderBrowserDialog();
            var result = browser.ShowDialog();
            if (result == DialogResult.OK) {
                PathToDataset = browser.SelectedPath;
            }
        }

        [RelayCommand]
        private void ChangePathToSavings() {
            var browser = new FolderBrowserDialog();
            var result = browser.ShowDialog();
            if (result == DialogResult.OK) {
                PathToSavings = browser.SelectedPath;
            }
        }
    }
}
