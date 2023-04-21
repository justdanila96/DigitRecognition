using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;

namespace DigitRecognition.ViewModels {
    internal partial class MainWindowViewModel : ObservableObject {

        public MainWindowViewModel() {

            MainPageViewModel.OpeningSettings += () => {
                Load("SettingsPage.xaml");
            };
        }

        [ObservableProperty]
        private Uri activeUri;
        
        [RelayCommand]
        private void Load(string pageName) {
            ActiveUri = new Uri($@"Pages\{pageName}", UriKind.Relative);
        }
    }
}
