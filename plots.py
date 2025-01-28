import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from data_loading import DataCore
import pandas as pd 
DC = DataCore()

def generate_plots(uploaded_path,selected_connector_number,language = "english"):
    # do tego tupla z wykresu 
    DC.load_excel(uploaded_path)
    DC.language = language
    figs = []
    dfs = []
    
    # Weibull distribution for each wavelength
    (fig, df) = generate_weibull_distribution_for_wavelengths(DC)    # 3 wykresy
    figs.extend(fig)
    dfs.extend(df)
    
    # 1550 vs 1310
    plot_vs = plot1550vs1310(DC)    # 1 wykres
    if plot_vs != None:
        (fig, df) = plot1550vs1310(DC)
        figs.append(fig)
        dfs.append(df)
    else:
        pass
    
    # Jumper mean
    (fig, df) = jumper_mean_plot_sorted(DC)
    figs.append(fig)
    dfs.append(df)
    
    # Jumper std
    (fig, df) = jumper_std_plot_sorted(DC)
    figs.append(fig)
    dfs.append(df)
    
    # RM286 IL mean
    (fig, df) = rm286_ilmean_plot(DC, selected_connector_number)
    figs.extend(fig)
    dfs.extend(df)
    
    # RM286 IL 97th
    (fig, df) = rm286_il97th_plot_filtered(DC,selected_connector_number)
    figs.extend(fig)
    dfs.extend(df)
    
    # Connector mean
    (fig, df) = connector_mean_plot_sorted(DC)
    figs.append(fig)
    dfs.append(df)
    
    # Connector std
    (fig, df) = connector_std_plot_sorted(DC)
    figs.append(fig)
    dfs.append(df)

    return [tuple(figs),tuple(dfs)]

def generate_tab_titles(lanuguage = "english"):
    if lanuguage == "english":
        titles = []
        titles_categories = []

        weibulls = []
        for wavelength in DC.wavelengths():
            weibulls.append(f"Weibull distribution {wavelength}")

        titles.append(weibulls)
        titles_categories.append("Weibull")

        if 1550 in DC.wavelengths() and 1310 in DC.wavelengths():
            titles += [["1550v1310"],]
            titles_categories.append("1550v1310")


        titles_categories.append("Jumper")
        titles += [[ "Jumper Mean", "Jumper Std"]]


        RMs =[]
        titles_categories.append("Random Mating")
        for wavelength in DC.wavelengths():
            RMs.append(f"RM286_IL_Mean {wavelength}")

        for wavelength in DC.wavelengths():
            RMs.append(f"RM286_IL_97th {wavelength}")

        titles += [RMs]
        titles_categories.append("Connectors")
        titles += [["Mean Connectors",
            "Connectors Std"]]
        
    if lanuguage == "polish":
        titles = []
        titles_categories = []

        weibulls = []
        for wavelength in DC.wavelengths():
            weibulls.append(f"Dystrybucja Weibull'a {wavelength}")

        titles.append(weibulls)
        titles_categories.append("Weibull")

        if 1550 in DC.wavelengths() and 1310 in DC.wavelengths():
            titles += [["1550v1310"],]
            titles_categories.append("1550v1310")


        titles_categories.append("Kable")
        titles += [[ "Kable średnie", "kable odchylenie"]]


        RMs =[]
        titles_categories.append("Random Mating")
        for wavelength in DC.wavelengths():
            RMs.append(f"RM286_IL_Mean {wavelength}")

        for wavelength in DC.wavelengths():
            RMs.append(f"RM286_IL_97th {wavelength}")

        titles += [RMs]
        titles_categories.append("Złącza")
        titles += [["Złącza średnie",
            "Złącza odchylenie"]]

    return titles,titles_categories

def plot_weibull_distribution(il_values, wavelength):
    """Plot histogram and Weibull fit for given IL values"""
    
    def weibull_distribution_fit(il_values):
        """Fit a Weibull distribution to IL data"""
        # Fit Weibull distribution to IL data
        shape, loc, scale = stats.weibull_min.fit(il_values, floc=0)  # Fix location at 0
        
        # Calculate the probability density function (PDF) values
        x = np.linspace(np.min(il_values), np.max(il_values), 1000)
        pdf_fitted = stats.weibull_min.pdf(x, shape, loc, scale)
        
        return shape, scale, x, pdf_fitted
    
    shape, scale, x, pdf_fitted = weibull_distribution_fit(il_values)

    
    # Calculate the mean of the IL values
    mean_il = np.mean(il_values)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    plt.hist(il_values, bins=30, density=True, alpha=0.6, color='g', label='Histogram of IL values')
    plt.plot(x, pdf_fitted, 'r-', label=f'Weibull fit\n(shape={shape:.2f}, scale={scale:.2f})')

    # Add mean value to the legend
    plt.axvline(mean_il, color='b', linestyle='dashed', linewidth=2, label=f'Mean = {mean_il:.2f}')
    if DC.language == "english":
        plt.title(f'Weibull Distribution Fit for Wavelength {wavelength} nm')
        plt.xlabel('IL values')
        plt.ylabel('Density')
    if DC.language == "polish":
        plt.title(f'Dopasowanie rozkładu Weibulla do długości fali {wavelength} nm')
        plt.xlabel('Wartości IL')
        plt.ylabel('Gęstość')
    plt.legend()
    plt.show()
    df = pd.DataFrame([[shape, scale, mean_il]],columns=["kształt weibull'a","skala weibull'a","średnie IL"])
    return (fig, df)

def generate_weibull_distribution_for_wavelengths(DC):
    """Generate Weibull distribution for each wavelength"""
    wavelength_IL_unfiltered = DC.IL_wavelengths()
    
    plots = []
    data  = []
    for wavelength, il_values in wavelength_IL_unfiltered.items():
        il_values_filtered = DC.filter_nan(il_values)  # Clean the data by removing NaNs
        plot,partial_data = plot_weibull_distribution(il_values_filtered, wavelength)
        plots.append(plot)
        data.append(partial_data)

    return (plots,data)

def plot1550vs1310(DC):
    """
    Tworzy wykres porównujący tłumienność wtrąceniową (IL) dla długości fal 1550 nm i 1310 nm.
    Korzysta z danych odczytanych za pomocą IL_reference_connectors().

    Parameters:
    ----------
    DC : DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    
    wavelength_ex = DC.wavelengths()[0]

    test_sheet = DC.IL_wavelength(wavelength_ex)

    #-----------------------------------------------------------------

    num_connectors = DC.n_connectors(test_sheet)    # number of connectors (prawidłowy)
    
    # Zaczytanie danych tłumienności dla wszystkich długości fal (IL dla konektorów)
    IL_data = DC.IL_dut_connectors()

    # Długości fal (muszą być dostępne w danych)
    wavelengths = DC.wavelengths()

    if 1310 not in wavelengths or 1550 not in wavelengths:
        return None
        raise ValueError("Długości fal 1310 nm lub 1550 nm nie znajdują się w dostępnych danych!")
    
    x = np.array([])
    y = np.array([])
    
    # Pobierz dane dla 1310 nm i 1550 nm
    idx_1310 = wavelengths.index(1310)
    idx_1550 = wavelengths.index(1550)
    
    for connector in range(0, num_connectors):
        start = 0
        end = num_connectors
        
        # Usuwanie NaN
        connector_data_cleaned = [
            np.array(connector_data, dtype=float)[~np.isnan(np.array(connector_data, dtype=float))] 
            for connector_data in IL_data[connector]
        ]
        
        # Obliczanie przy wybranej długości fali
        for wavelength in range(0, len(wavelengths)):
            connector_data_wavelength = connector_data_cleaned[start:end]
            
            # Filter out empty arrays
            connector_data_wavelength = [arr for arr in connector_data_wavelength if arr.size > 0]

            # Concatenate the remaining arrays into a single 1D array
            connector_data_wavelength = np.concatenate(connector_data_wavelength)
            
            if wavelength == idx_1310:
                x = np.append(x, connector_data_wavelength)
            elif wavelength == idx_1550:
                y = np.append(y, connector_data_wavelength)
            
            start += num_connectors
            end += num_connectors
    
    x_cleaned = x
    y_cleaned = y
    
    x_clean = np.array(x_cleaned)
    y_clean = np.array(y_cleaned)

    # Tworzenie wykresu
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(x_cleaned, y_cleaned, color='blue', label='Data points')
    
    # # Dopasuj linię trendu i narysuj
    z = np.polyfit(x_clean, y_clean, 1)  # Dopasowanie liniowe (stopień 1)
    p = np.poly1d(z)
    plt.plot(x_clean, p(x_clean), color='red', label=f'Trendline: y = {z[0]:.2f}x + {z[1]:.2f}')

    # # Oblicz RMSE i R^2
    y_pred = p(x_clean)  # Przewidywane wartości z linii trendu
    rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))
    r2 = 1 - (np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2))

    # Narysuj funkcję IL1550 = IL1310 * 0.78
    il_1550 = x_clean * 0.78
    plt.plot(x_clean, il_1550, color='green', linestyle='--', label='IL1550 = IL1310 × 0.78')
    
    # # Oś symetryczna; Symmetric axis
    plt.axis('equal')
    axis_min = min(np.min(x_clean)-.02, np.min(y_clean)-.02)
    axis_max = max(np.max(x_clean)+.02, np.max(y_clean)+.02)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.gca().set_aspect('equal', adjustable='box')

    # Dodaj etykiety, legendę i tytuł
    plt.xlabel('1310')
    plt.ylabel('1550')
    plt.grid()
    plt.legend(title=f'RMSE = {rmse:.4f}, R² = {r2:.4f}')
    plt.title('1550 vs 1310')

    # Pokaż wykres
    plt.show()
    df = pd.DataFrame([[rmse, r2, z[0], z[1]]],columns=["RMSE","R^2","a","b"])
    
    return (fig, df)

def rm286_ilmean_plot(data_core: DataCore, selected_connector_number):
    """
    Plot histograms of IL mean values for all wavelengths with statistics.

    Parameters:
    ----------
    data_core : DataCore
        An instance of the DataCore class with loaded Excel data.
    """
    wavelengths = data_core.wavelengths()
    wave_combinations_IL_unfiltered = data_core.jumper_combinations_all_wavelengths(selected_connector_number)
    wave_combinations_IL = data_core.map_dict(data_core.filter_nan, wave_combinations_IL_unfiltered)
    wave_combinations_IL_mean = DC.map_dict(lambda arr : np.mean(arr,axis=1), wave_combinations_IL)
    
    plots = []
    stat=[]
    
    for wavelength in wavelengths:
        # Get mean IL values for all connectors for the given wavelength

        
        data = wave_combinations_IL_mean[wavelength]


        # Calculate statistics
        mean = np.mean(data)
        std_dev = np.std(data)
        df = pd.DataFrame([[mean, std_dev]],columns=["mean","std_dev"])
        stat.append(df)

        # Plot histogram
        fig = plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, color='blue', alpha=0.7, label='Data distribution')

        # Plot mean
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')

        # Plot standard deviation lines
        plt.axvline(mean - std_dev, color='green', linestyle='--', label=f'Mean - 1 SD = {mean - std_dev:.2f}')
        plt.axvline(mean + std_dev, color='green', linestyle='--', label=f'Mean + 1 SD = {mean + std_dev:.2f}')

        # Add additional vertical lines at specific values
        plt.axvline(0.1, color='purple', linestyle='-', label='Threshold = 0.1')
        plt.axvline(0.12, color='orange', linestyle='-', label='Threshold = 0.12')

        # Add legend, title, and labels
        plt.legend()
        if DC.language == "english":
            plt.title(f'Histogram of IL Mean Values (Wavelength = {wavelength} nm)')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        if DC.language == "polish":
            plt.title(f'Histogram średnich wartości IL (długość fali = {wavelength} nm)')
            plt.xlabel('Wartość')
            plt.ylabel('Częstotliwość')

        plt.grid()

        # Show plot
        plt.show()
        plots.append(fig)
        
    return (plots, stat)


def rm286_il97th_plot_filtered(data_core: DataCore, selected_connector_number):
    """
    Plot histograms of filtered IL 97th percentile values for all wavelengths.
    Thresholds are dynamically adjusted based on mean and standard deviation.

    Parameters:
    ----------
    data_core : DataCore
        An instance of the DataCore class with loaded Excel data.
    """
    wavelengths = data_core.wavelengths()
    plots = []
    stat = []

    for wavelength in wavelengths:
        # Retrieve 97th percentile values for all connectors
        wave_combinations_IL_unfiltered = data_core.jumper_combinations_all_wavelengths(selected_connector_number)
        wave_combinations_IL = data_core.map_dict(data_core.filter_nan, wave_combinations_IL_unfiltered)

        # Calculate 97th percentile for each connector combination
        percentile_97th_values = [
            np.percentile(combination, 97) for combination in wave_combinations_IL[wavelength]
        ]

        # Convert the result to a NumPy array for statistical calculations
        data = np.array(percentile_97th_values)

        # Calculate statistics
        mean = np.mean(data)
        std_dev = np.std(data)

        # Define dynamic thresholds
        lower_threshold = max(mean - std_dev, 0)  # Ensure lower threshold is non-negative
        upper_threshold = mean + std_dev


        df = pd.DataFrame([[mean, std_dev, lower_threshold, upper_threshold]],columns=["mean","std_dev","lower_threshold","upper_threshold"])
        stat.append(df)

        # Plot histogram
        fig = plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, color='blue', alpha=0.7, label='Data distribution')

        # Plot mean
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')

        # Plot standard deviation lines
        plt.axvline(mean - std_dev, color='green', linestyle='--', label=f'Mean - 1 SD = {mean - std_dev:.2f}')
        plt.axvline(mean + std_dev, color='green', linestyle='--', label=f'Mean + 1 SD = {mean + std_dev:.2f}')

        # Plot dynamic thresholds
        plt.axvline(lower_threshold, color='purple', linestyle='-', label=f'Lower Threshold = {lower_threshold:.2f}')
        plt.axvline(upper_threshold, color='orange', linestyle='-', label=f'Upper Threshold = {upper_threshold:.2f}')

        # Add legend, title, and labels
        plt.legend()
        if DC.language == "english":
            plt.title(f'Filtered Histogram of IL 97th Percentile Values (Wavelength = {wavelength} nm)')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        if DC.language == "polish":
            plt.title(f'Przefiltrowany histogram wartości IL 97 percentyla (długość fali = {wavelength} nm)')
            plt.xlabel('Wartość')
            plt.ylabel('Częstotliwość')
        plt.grid()

        # Show plot
        plt.show()
        
        plots.append(fig)
        
    
    return (plots, stat)

def jumper_mean_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający średnią wartość dla różnych jumperów i długości fal, 
    z uporządkowaną osią X tak, aby na początku (x=1) był numer jumpera z najniższą mean value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
        
    n_choices: int
        Liczba jumperów, które chcemy uwzględnić w wykresie (np. wybierając 1, będziemy patrzeć na pojedyncze jumpers).
    """
    
    # Ładowanie danych
    # To nigdy nie miało sensu i działało (błędnie) przez błąd w definicji danych
    # zmienić na DataCore.IL_jumpers_wavelengths
    wavelength_IL_combinations = DC.IL_jumpers_wavelengths()
    wavelengths = list(wavelength_IL_combinations.keys())
    
    # Przygotowanie danych jumperów
    jumper_numbers = np.array(range(1, DC.n_jumpers(DC.IL_wavelength(wavelengths[0])) + 1))
    
    # Wybór długości fali do sortowania
    wavelength_to_sort = wavelengths[0]
    IL_data_to_sort = wavelength_IL_combinations[wavelength_to_sort]
    
    # Sprawdzanie i usuwanie wartości NaN w danych
    IL_data_to_sort_clean = [
        np.array(jumper_data, dtype=float)[~np.isnan(np.array(jumper_data, dtype=float))]  # Rzutowanie na float i usuwanie NaN
        for jumper_data in IL_data_to_sort
    ]
    
    # Obliczamy średnie wartości dla jumperów przy wybranej długości fali
    mean_values_to_sort = np.array([np.mean(data) if len(data) > 0 else np.nan for data in IL_data_to_sort_clean])
    
    # Sortowanie jumperów na podstawie średnich wartości dla wybranej długości fali
    sorted_indices = np.argsort(mean_values_to_sort)
    sorted_jumper_numbers = jumper_numbers[sorted_indices]
    
    # Tworzymy wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.Set2(range(0, len(wavelength_IL_combinations)))

    all_mean_values = []

    # Przechodzimy po długościach fal i rysujemy wykres dla posortowanych jumperów
    for idx, (wavelength, IL_data) in enumerate(wavelength_IL_combinations.items()):
        # Obliczamy średnie wartości dla wszystkich jumperów
        mean_values = [
            np.mean(np.array(jumper_data, dtype=float)[~np.isnan(np.array(jumper_data, dtype=float))]) 
            if len(jumper_data) > 0 else np.nan 
            for jumper_data in IL_data
        ]
        
        # Sortowanie średnich wartości zgodnie z kolejnością jumperów
        sorted_mean_values = np.array(mean_values)[sorted_indices]
        all_mean_values.append(list(sorted_mean_values))

        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_jumper_numbers) + 1),  # Indeksy na osi X od 1 do n
            sorted_mean_values, 
            label=f'Mean (λ={wavelength} nm)', 
            color=colors[idx], 
            linestyle='-', 
            marker='o'
        )
    
    # Ustawienie etykiet na osi X zgodnie z numerami jumperów
    ax.set_xticks(range(1, len(sorted_jumper_numbers) + 1))
    ax.set_xticklabels(sorted_jumper_numbers)  # Używamy posortowanych numerów jumperów jako etykiet
    
    # Ustawienia wykresu
    if DC.language == "english":
        ax.set_xlabel('Jumper Number (Ordered by Mean for λ={:.0f} nm)'.format(wavelength_to_sort))
        ax.set_ylabel('Mean IL Value')
        ax.set_title('Sorted Mean IL Values for Jumpers Across Different Wavelengths')
    
    if DC.language == "polish":
        ax.set_xlabel('Numer Kabla (Posortowany po średniej dla λ={:.0f} nm)'.format(wavelength_to_sort))
        ax.set_ylabel('Średnia wartość IL')
        ax.set_title('Posortowane średnie wartości IL dla kabli o różnych długościach fali')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.show()    
    df = pd.DataFrame(all_mean_values,columns=[f"jumper {i}" for i in sorted_jumper_numbers])
    return (fig, df)

def jumper_std_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający odchylenie standardowe dla różnych jumperów i długości fal, 
    z uporządkowaną osią X tak, aby na początku (x=1) był numer jumpera z najniższym mean value.
    (Mozna zmienic na sortowanie pod odchyleniu standardowym)
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
        
    n_choices: int
        Liczba jumperów, które chcemy uwzględnić w wykresie (np. wybierając 1, będziemy patrzeć na pojedyncze jumpers).
    """
    # Ładowanie danych
    wavelength_IL_combinations = DC.IL_jumpers_wavelengths()
    wavelengths = list(wavelength_IL_combinations.keys())
    
    # Przygotowanie danych jumperów
    jumper_numbers = np.array(range(1, DC.n_jumpers(DC.IL_wavelength(wavelengths[0])) + 1))
    
    # Wybór długości fali do sortowania
    wavelength_to_sort = wavelengths[0]
    IL_data_to_sort = wavelength_IL_combinations[wavelength_to_sort]
    
    # Sprawdzanie i usuwanie wartości NaN w danych
    IL_data_to_sort_clean = [
        np.array(jumper_data, dtype=float)[~np.isnan(np.array(jumper_data, dtype=float))]  # Rzutowanie na float i usuwanie NaN
        for jumper_data in IL_data_to_sort
    ]
    
    # # Obliczamy odchylenia standardowe dla jumperów przy wybranej długości fali
    # std_values_to_sort = np.array([
    #     np.std(data) if len(data) > 0 else np.nan 
    #     for data in IL_data_to_sort_clean
    # ])
    
    # # Sortowanie jumperów na podstawie odchyleń standardowych dla wybranej długości fali
    # sorted_indices = np.argsort(std_values_to_sort)
    # sorted_jumper_numbers = jumper_numbers[sorted_indices]
    
    # Obliczamy średnie wartości dla jumperów przy wybranej długości fali
    mean_values_to_sort = np.array([np.mean(data) if len(data) > 0 else np.nan for data in IL_data_to_sort_clean])
    
    # Sortowanie jumperów na podstawie średnich wartości dla wybranej długości fali
    sorted_indices = np.argsort(mean_values_to_sort)
    sorted_jumper_numbers = jumper_numbers[sorted_indices]
    
    # Tworzymy wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.Set2(range(0, len(wavelength_IL_combinations)))
    
    all_std_values = []
    all_mean_values = []

    # Przechodzimy po długościach fal i rysujemy wykres dla posortowanych jumperów
    for idx, (wavelength, IL_data) in enumerate(wavelength_IL_combinations.items()):
        # Sprawdzanie i usuwanie wartości NaN w danych dla bieżącej długości fali
        IL_data_clean = [
            np.array(jumper_data, dtype=float)[~np.isnan(np.array(jumper_data, dtype=float))]
            for jumper_data in IL_data
        ]
        
        # Obliczamy odchylenia standardowe dla wszystkich jumperów
        std_values = [
            np.std(data) if len(data) > 0 else np.nan 
            for data in IL_data_clean
        ]
        
        mean_values = [
            np.mean(np.array(jumper_data, dtype=float)[~np.isnan(np.array(jumper_data, dtype=float))]) 
            if len(jumper_data) > 0 else np.nan 
            for jumper_data in IL_data
        ]
        
        # Sortowanie średnich wartości zgodnie z kolejnością jumperów
        sorted_mean_values = np.array(mean_values)[sorted_indices]
        all_mean_values.append(list(sorted_mean_values))
        
        # Sortowanie odchyleń standardowych zgodnie z kolejnością jumperów
        sorted_std_values = np.array(std_values)[sorted_indices]

        all_std_values.append(list(sorted_std_values))


        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_jumper_numbers) + 1),  # Indeksy na osi X od 1 do n
            sorted_std_values, 
            label=f'Standard Deviation (λ={wavelength} nm)', 
            color=colors[idx], 
            linestyle='--', 
            marker='x'
        )
    
    # Ustawienie etykiet na osi X zgodnie z numerami jumperów
    ax.set_xticks(range(1, len(sorted_jumper_numbers) + 1))
    ax.set_xticklabels(sorted_jumper_numbers)  # Używamy posortowanych numerów jumperów jako etykiet
    
    # Ustawienia wykresu
    if DC.language == "english":
        ax.set_xlabel('Jumper Number (Ordered by Standard Deviation for λ={:.0f} nm)'.format(wavelength_to_sort))
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Sorted Standard Deviation of IL Values for Jumpers Across Different Wavelengths')
    if DC.language == "polish":
        ax.set_xlabel('Numer Kabla (Posortowany odchyleniem standardowym dla  λ={:.0f} nm)'.format(wavelength_to_sort))
        ax.set_ylabel('Odchylenie Standardowe')
        ax.set_title('Posortowane odchylenie standardowe wartości IL dla zworek o różnych długościach fali')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.show()
    
    df = pd.DataFrame(all_std_values,columns=[f"jumper {i}" for i in sorted_jumper_numbers])
    return (fig, df)

def connector_std_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający odchylenie standardowe IL dla różnych connectorów i długości fal,
    z uporządkowaną osią X tak, aby na początku (x=1) był numer connectora z najniższym std value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    wavelength_ex = DC.wavelengths()[0]

    test_sheet = DC.IL_wavelength(wavelength_ex)

    num_connectors = DC.n_connectors(test_sheet)    # number of connectors (prawidłowy)
    
    # Ładowanie danych dla connectorów
    wavelengths = DC.wavelengths()
    IL_data = DC.IL_dut_connectors()

    # Lista do przechowywania wyników podzielonych na długości fal
    grouped_by_wavelength = [[] for _ in wavelengths]
    
    for connector in range(0, num_connectors):
        start = 0
        end = num_connectors
        
        # Usuwanie NaN przed obliczaniem średniej
        connector_data_cleaned = [
            np.array(connector_data, dtype=float)[~np.isnan(np.array(connector_data, dtype=float))] 
            for connector_data in IL_data[connector]
        ]
        
        # Obliczanie wartości średnich dla connectora przy wybranej długości fali
        for wavelength in range(0, len(wavelengths)):
            connector_data_wavelength = connector_data_cleaned[start:end]
            std_for_wavelength = np.array([np.std(connector_data) if len(connector_data) > 0 else np.nan for connector_data in connector_data_wavelength])
            std_value = np.nanstd(std_for_wavelength)
            
            # Dodanie wyniku do odpowiedniej grupy długości fal
            grouped_by_wavelength[wavelength].append(std_value)
            
            start += num_connectors
            end += num_connectors

    # Konwersja wyników do macierzy NumPy
    std_values_array = np.array(grouped_by_wavelength).T

    # Sortowanie connectorów na podstawie wartości średnich dla wybranej długości fali
    wavelength_to_sort = 0
    sorted_indices = np.argsort(std_values_array[:, wavelength_to_sort])
    sorted_connector_numbers = np.arange(1, num_connectors + 1)[sorted_indices]

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))

    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.Set2(range(0, len(wavelengths)))

    all_std_values = []

    # Rysowanie wykresu dla każdej długości fali
    for idx, wavelength in enumerate(wavelengths):
        # Pobranie średnich wartości dla danej długości fali
        std_values = std_values_array[:, idx]

        # Sortowanie średnich wartości zgodnie z posortowanymi connectorami
        sorted_std_values = std_values[sorted_indices]
        all_std_values.append(sorted_std_values)

        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_connector_numbers) + 1),  # Indeksy na osi X
            sorted_std_values,
            label=f'Std (\u03bb={wavelength} nm)',
            color=colors[idx],
            linestyle='--',
            marker='x'
        )

    # Ustawienia osi X
    ax.set_xticks(range(1, len(sorted_connector_numbers) + 1))
    ax.set_xticklabels(sorted_connector_numbers)  # Etykiety to posortowane numery connectorów

    # Ustawienia wykresu
    if DC.language == "english":
        ax.set_xlabel(f'Connector Number (Ordered by Mean for \u03bb={wavelengths[0]} nm)')
        ax.set_ylabel('Std of IL')
        ax.set_title('Sorted Std of IL Values for Connectors Across Different Wavelengths')
    if DC.language == "polish":
        ax.set_xlabel(f'Numer złącza (Posortowany po średniej dla \u03bb={wavelengths[0]} nm)')
        ax.set_ylabel('Standardowe odchylenie')
        ax.set_title('Posortowane odchylenie standardowe wartości IL dla złączy o różnych długościach fali')

    ax.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Przygotowanie DataFrame do zwrócenia
    df = pd.DataFrame(
        np.array(all_std_values).T,  # Transponujemy, aby connectorzy byli w wierszach
        columns=[f"Wavelength {wavelength} nm" for wavelength in wavelengths]
    )
    df.index = [f"Connector {connector}" for connector in sorted_connector_numbers]

    df = pd.DataFrame(all_std_values, columns=[f"connector {i}" for i in sorted_connector_numbers])
    
    return (fig, df)

def connector_mean_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający średnią IL dla różnych connectorów i długości fal,
    z uporządkowaną osią X tak, aby na początku (x=1) był numer connectora z najniższą średnią wartością IL.

    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    
    wavelength_ex = DC.wavelengths()[0]

    test_sheet = DC.IL_wavelength(wavelength_ex)

    #-----------------------------------------------------------------

    num_connectors = DC.n_connectors(test_sheet)    # number of connectors (prawidłowy)
    
    # Ładowanie danych dla connectorów
    wavelengths = DC.wavelengths()
    IL_data = DC.IL_dut_connectors()

    # Lista do przechowywania wyników podzielonych na długości fal
    grouped_by_wavelength = [[] for _ in wavelengths]
    
    for connector in range(0, num_connectors):
        start = 0
        end = num_connectors
        
        # Usuwanie NaN przed obliczaniem średniej
        connector_data_cleaned = [
            np.array(connector_data, dtype=float)[~np.isnan(np.array(connector_data, dtype=float))] 
            for connector_data in IL_data[connector]
        ]
        
        # Obliczanie wartości średnich dla connectora przy wybranej długości fali
        for wavelength in range(0, len(wavelengths)):
            connector_data_wavelength = connector_data_cleaned[start:end]
            mean_for_wavelength = np.array([np.mean(connector_data) if len(connector_data) > 0 else np.nan for connector_data in connector_data_wavelength])
            mean_value = np.nanmean(mean_for_wavelength)
            
            # Dodanie wyniku do odpowiedniej grupy długości fal
            grouped_by_wavelength[wavelength].append(mean_value)
            
            start += num_connectors
            end += num_connectors

    # Konwersja wyników do macierzy NumPy
    mean_values_array = np.array(grouped_by_wavelength).T

    # Sortowanie connectorów na podstawie wartości średnich dla wybranej długości fali
    wavelength_to_sort = 0
    sorted_indices = np.argsort(mean_values_array[:, wavelength_to_sort])
    sorted_connector_numbers = np.arange(1, num_connectors + 1)[sorted_indices]

    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))

    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.Set2(range(0, len(wavelengths)))

    all_mean_values = []

    # Rysowanie wykresu dla każdej długości fali
    for idx, wavelength in enumerate(wavelengths):
        # Pobranie średnich wartości dla danej długości fali
        mean_values = mean_values_array[:, idx]

        # Sortowanie średnich wartości zgodnie z posortowanymi connectorami
        sorted_mean_values = mean_values[sorted_indices]
        all_mean_values.append(sorted_mean_values)

        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_connector_numbers) + 1),  # Indeksy na osi X
            sorted_mean_values,
            label=f'Mean (\u03bb={wavelength} nm)',
            color=colors[idx],
            linestyle='--',
            marker='x'
        )

    # Ustawienia osi X
    ax.set_xticks(range(1, len(sorted_connector_numbers) + 1))
    ax.set_xticklabels(sorted_connector_numbers)  # Etykiety to posortowane numery connectorów

    # Ustawienia wykresu

    if DC.language == "english":
        ax.set_xlabel(f'Connector Number (Ordered by Mean for \u03bb={wavelengths[0]} nm)')
        ax.set_ylabel('Mean of IL')
        ax.set_title('Sorted Mean of IL Values for Connectors Across Different Wavelengths')
    if DC.language == "polish":
        ax.set_xlabel(f'Numer złącza (Posortowany po średniej dla \u03bb={wavelengths[0]} nm)')
        ax.set_ylabel('Średnia IL')
        ax.set_title('Posortowane średnie wartości IL dla złączy o różnych długościach fali')
    ax.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Przygotowanie DataFrame do zwrócenia
    df = pd.DataFrame(
        np.array(all_mean_values).T,  # Transponujemy, aby connectorzy byli w wierszach
        columns=[f"Wavelength {wavelength} nm" for wavelength in wavelengths]
    )
    df.index = [f"Connector {connector}" for connector in sorted_connector_numbers]

    df = pd.DataFrame(all_mean_values, columns=[f"connector {i}" for i in sorted_connector_numbers])
    
    return (fig, df)