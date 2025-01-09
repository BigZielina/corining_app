import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import itertools as it
import xlsxwriter
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
import data_loading
import scipy.stats as stats
from data_loading import DataCore

DC = DataCore()

def generate_plots(uploaded_path):
    # do tego tupla z wykresu 
    DC.load_excel(uploaded_path)
    figs = []
    figs.extend(generate_weibull_distribution_for_wavelengths(DC))    # 3 wykresy
    plot_vs = plot1550vs1310(DC)    # 1 wykres
    if plot_vs != None:
        figs.append(plot_vs)

    figs.append(jumper_mean_plot_sorted(DC, n_choices=1))
    figs.append(jumper_std_plot_sorted(DC, n_choices=1))
    figs.extend(rm286_ilmean_plot(DC))
    figs.extend(rm286_il97th_plot_filtered(DC))
    figs.append(connector_mean_plot_sorted(DC))
    figs.append(connector_std_plot_sorted(DC))
    
    return tuple(figs)

def generate_tab_titles():
    titles = []
    for wavelength in DC.wavelengths():
    titles.append(f"Weibull distribution {wavelength}")
     titles += ["1550v1310d", "Jumper Mean", "Jumper Std"]
    for wavelength in DC.wavelengths():
    titles.append(f"RM286_IL_Mean {wavelength}")
        titles.append(f"RM286_IL_97th {wavelength}")

    titles += ["Mean Connectors",
        "Connectors Std"]

    return titles

def connector_ilstd_plot(wavelengths, data_core = DC):

    connector_data = data_core.IL_reference_connectors()
    num_connectors = connector_data.shape[0]  # Number of connectors
    connector_std = {wavelength: [] for wavelength in wavelengths}

    for wavelength in wavelengths:

        for connector_index in range(num_connectors):
            il_values = connector_data[connector_index, :]
            il_values_filtered = data_core.filter_nan(il_values)
            connector_std[wavelength].append(il_values_filtered.std())


    connector_order = range(1, num_connectors + 1)


    plt.figure(figsize=(10, 6))
    for wavelength in wavelengths:
        plt.plot(connector_order, connector_std[wavelength], marker='o', linestyle='-', label=f'{wavelength} nm')


    plt.xticks(ticks=range(1, num_connectors + 1), labels=connector_order)


    plt.xlabel('Connector Ranking')
    plt.ylabel('IL Std')
    plt.grid(axis='y')
    plt.legend()
    plt.title('Connector Ranking vs IL Std')

    plt.show()



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
    
    plt.title(f'Weibull Distribution Fit for Wavelength {wavelength} nm')
    plt.xlabel('IL Values')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    return fig

def generate_weibull_distribution_for_wavelengths(DC):
    """Generate Weibull distribution for each wavelength"""
    wavelength_IL_unfiltered = DC.IL_wavelengths()
    
    results = []

    for wavelength, il_values in wavelength_IL_unfiltered.items():
        il_values_filtered = DC.filter_nan(il_values)  # Clean the data by removing NaNs
        results.append(plot_weibull_distribution(il_values_filtered, wavelength))

    return results

def plot1550vs1310(DC):
    """
    Tworzy wykres porównujący tłumienność wtrąceniową (IL) dla długości fal 1550 nm i 1310 nm.
    Korzysta z danych odczytanych za pomocą IL_reference_connectors().

    Parameters:
    ----------
    DC : DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    
    # Zaczytanie danych tłumienności dla wszystkich długości fal (IL dla konektorów)
    IL_data = DC.IL_reference_connectors()

    # Długości fal (muszą być dostępne w danych)
    wavelengths = DC.wavelengths()

    if 1310 not in wavelengths or 1550 not in wavelengths:
        return
        raise ValueError("Długości fal 1310 nm lub 1550 nm nie znajdują się w dostępnych danych!")

    # Pobierz dane dla 1310 nm i 1550 nm
    idx_1310 = wavelengths.index(1310)
    idx_1550 = wavelengths.index(1550)

    x = np.array(IL_data[idx_1310], dtype=float)  # Dane dla 1310 nm
    y = np.array(IL_data[idx_1550], dtype=float)  # Dane dla 1550 nm

    # Usuń wartości NaN z obu zestawów danych
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Stwórz wykres scatter
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x_clean, y_clean, color='blue', label='Data points', s=10)

    # Dopasuj linię trendu i narysuj
    z = np.polyfit(x_clean, y_clean, 1)  # Dopasowanie liniowe (stopień 1)
    p = np.poly1d(z)
    plt.plot(x_clean, p(x_clean), color='red', label=f'Trendline: y = {z[0]:.2f}x + {z[1]:.2f}')

    # Oblicz RMSE i R^2
    y_pred = p(x_clean)  # Przewidywane wartości z linii trendu
    rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))
    r2 = 1 - (np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2))

    # Narysuj funkcję IL1550 = IL1310 * 0.78
    il_1550 = x_clean * 0.78
    plt.plot(x_clean, il_1550, color='green', linestyle='--', label='IL1550 = IL1310 × 0.78')

    # Oś symetryczna
    axis_min = min(np.min(x_clean)-.02, np.min(y_clean)-.02)
    axis_max = max(np.max(x_clean)+.02, np.max(y_clean)+.02)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    # Dodaj etykiety, legendę i tytuł
    plt.xlabel('1310')
    plt.ylabel('1550')
    plt.grid()
    plt.legend(title=f'RMSE = {rmse:.4f}, R² = {r2:.4f}')
    plt.title('1550 vs 1310')

    # Pokaż wykres
    plt.show()
    return fig

def rm286_ilmean_plot(data_core: DataCore):
    """
    Plot histograms of IL mean values for all wavelengths with statistics.

    Parameters:
    ----------
    data_core : DataCore
        An instance of the DataCore class with loaded Excel data.
    """
    wavelengths = data_core.wavelengths()
    plots = []
    
    for wavelength in wavelengths:
        # Get mean IL values for all connectors for the given wavelength
        data = data_core.IL_wavelength(wavelength).iloc[2:, 2:].to_numpy().astype(float)
        data = data[~np.isnan(data)]  # Filter NaN values

        # Calculate statistics
        mean = np.mean(data)
        std_dev = np.std(data)

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
        plt.title(f'Histogram of IL Mean Values (Wavelength = {wavelength} nm)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid()

        # Show plot
        plt.show()
        plots.append(fig)
        
    return plots

def rm286_il97th_plot_filtered(data_core: DataCore):
    """
    Plot histograms of filtered IL 97th percentile values for all wavelengths.

    Parameters:
    ----------
    data_core : DataCore
        An instance of the DataCore class with loaded Excel data.
    """
    wavelengths = data_core.wavelengths()
    plots = []

    for wavelength in wavelengths:
        # Retrieve 97th percentile values for all connectors
        wave_combinations_IL_unfiltered = data_core.jumper_combinations_all_wavelengths(4)
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

        # Plot histogram
        fig = plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, color='blue', alpha=0.7, label='Data distribution')

        # Plot mean
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')

        # Plot standard deviation lines
        plt.axvline(mean - std_dev, color='green', linestyle='--', label=f'Mean - 1 SD = {mean - std_dev:.2f}')
        plt.axvline(mean + std_dev, color='green', linestyle='--', label=f'Mean + 1 SD = {mean + std_dev:.2f}')

        # Add additional vertical lines at specific values
        plt.axvline(0.15, color='purple', linestyle='-', label='Threshold = 0.15')
        plt.axvline(0.25, color='orange', linestyle='-', label='Threshold = 0.25')

        # Add legend, title, and labels
        plt.legend()
        plt.title(f'Filtered Histogram of IL 97th Percentile Values (Wavelength = {wavelength} nm)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid()

        # Show plot
        plt.show()
        
        plots.append(fig)
        
    return plots

def jumper_mean_plot_sorted(DC, n_choices=1):
    """
    Tworzy wykres przedstawiający średnią wartość dla różnych jumperów i długości fal, 
    z uporządkowaną osią X tak, aby na początku (x=1) był numer jumpera z najniższą mean value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
        
    n_choices: int
        Liczba jumperów, które chcemy uwzględnić w wykresie (np. wybierając 1, będziemy patrzeć na pojedyncze jumpers).
    """
    
    # Ładowanie danych
    wavelength_IL_combinations = DC.jumper_combinations_all_wavelengths(n_choices)
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
    colors = plt.cm.viridis(np.linspace(0, 1, len(wavelength_IL_combinations)))
    
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
    ax.set_xlabel('Jumper Number (Ordered by Mean for λ={:.0f} nm)'.format(wavelength_to_sort))
    ax.set_ylabel('Mean IL Value')
    ax.set_title('Sorted Mean IL Values for Jumpers Across Different Wavelengths')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.show()    
    return fig

def jumper_std_plot_sorted(DC, n_choices=1):
    """
    Tworzy wykres przedstawiający odchylenie standardowe dla różnych jumperów i długości fal, 
    z uporządkowaną osią X tak, aby na początku (x=1) był numer jumpera z najniższym std value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
        
    n_choices: int
        Liczba jumperów, które chcemy uwzględnić w wykresie (np. wybierając 1, będziemy patrzeć na pojedyncze jumpers).
    """
    # Ładowanie danych
    wavelength_IL_combinations = DC.jumper_combinations_all_wavelengths(n_choices)
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
    
    # Obliczamy odchylenia standardowe dla jumperów przy wybranej długości fali
    std_values_to_sort = np.array([
        np.std(data) if len(data) > 0 else np.nan 
        for data in IL_data_to_sort_clean
    ])
    
    # Sortowanie jumperów na podstawie odchyleń standardowych dla wybranej długości fali
    sorted_indices = np.argsort(std_values_to_sort)
    sorted_jumper_numbers = jumper_numbers[sorted_indices]
    
    # Tworzymy wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.viridis(np.linspace(0, 1, len(wavelength_IL_combinations)))
    
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
        
        # Sortowanie odchyleń standardowych zgodnie z kolejnością jumperów
        sorted_std_values = np.array(std_values)[sorted_indices]
        
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
    ax.set_xlabel('Jumper Number (Ordered by Standard Deviation for λ={:.0f} nm)'.format(wavelength_to_sort))
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Sorted Standard Deviation of IL Values for Jumpers Across Different Wavelengths')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.show()
    
    return fig

def connector_std_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający odchylenie standardowe IL dla różnych connectorów i długości fal,
    z uporządkowaną osią X tak, aby na początku (x=1) był numer connectora z najniższym std value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    # Ładowanie danych dla connectorów
    wavelengths = DC.wavelengths()
    IL_data = DC.IL_reference_connectors()
    connector_numbers = np.array(range(1, len(IL_data[0]) + 1))  # Numery connectorów
    
    # Wybór pierwszej długości fali do sortowania
    wavelength_to_sort = wavelengths[0]
    IL_data_to_sort = IL_data[0]  # Zakładamy, że dane są posortowane według długości fal w IL_data
    
    # Obliczanie wartości odchyleń standardowych dla connectorów przy wybranej długości fali
    std_values_to_sort = np.array([np.std(IL_data_to_sort[connector - 1]) for connector in connector_numbers])
    
    # Sortowanie connectorów na podstawie std values
    sorted_indices = np.argsort(std_values_to_sort)
    sorted_connector_numbers = connector_numbers[sorted_indices]
    
    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.viridis(np.linspace(0, 1, len(wavelengths)))
    
    # Rysowanie wykresu dla każdej długości fali
    for idx, wavelength in enumerate(wavelengths):
        # Obliczanie std values dla wszystkich connectorów
        std_values = np.array([np.std(IL_data[idx][connector - 1]) for connector in connector_numbers])
        
        # Sortowanie std values zgodnie z posortowanymi connectorami
        sorted_std_values = std_values[sorted_indices]
        
        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_connector_numbers) + 1),  # Indeksy na osi X
            sorted_std_values,
            label=f'Standard Deviation (λ={wavelength} nm)',
            color=colors[idx],
            linestyle='--',
            marker='x'
        )
    
    # Ustawienie etykiet na osi X zgodnie z numerami connectorów
    ax.set_xticks(range(1, len(sorted_connector_numbers) + 1))
    ax.set_xticklabels(sorted_connector_numbers)  # Etykiety to posortowane numery connectorów
    
    # Ustawienia wykresu
    ax.set_xlabel(f'Connector Number (Ordered by Standard Deviation for λ={wavelength_to_sort} nm)')
    ax.set_ylabel('Standard Deviation of IL')
    ax.set_title('Sorted Standard Deviation of IL Values for Connectors Across Different Wavelengths')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.tight_layout()  # Poprawienie układu wykresu
    plt.show()
    
    return fig

def connector_mean_plot_sorted(DC):
    """
    Tworzy wykres przedstawiający odchylenie standardowe IL dla różnych connectorów i długości fal,
    z uporządkowaną osią X tak, aby na początku (x=1) był numer connectora z najniższym std value.
    
    DC: DataCore
        Obiekt klasy DataCore zawierający dane.
    """
    # Ładowanie danych dla connectorów
    wavelengths = DC.wavelengths()
    IL_data = DC.IL_reference_connectors()
    connector_numbers = np.array(range(1, len(IL_data[0]) + 1))  # Numery connectorów
    
    # Wybór pierwszej długości fali do sortowania
    wavelength_to_sort = wavelengths[0]
    IL_data_to_sort = IL_data[0]  # Zakładamy, że dane są posortowane według długości fal w IL_data
    
    # Obliczanie wartości odchyleń standardowych dla connectorów przy wybranej długości fali
    mean_values = np.array([np.mean(IL_data_to_sort[connector - 1]) for connector in connector_numbers])
    
    # Sortowanie connectorów na podstawie std values
    sorted_indices = np.argsort(mean_values)
    sorted_connector_numbers = connector_numbers[sorted_indices]
    
    # Tworzenie wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Przygotowanie kolorów dla różnych długości fal
    colors = plt.cm.viridis(np.linspace(0, 1, len(wavelengths)))
    
    # Rysowanie wykresu dla każdej długości fali
    for idx, wavelength in enumerate(wavelengths):
        # Obliczanie std values dla wszystkich connectorów
        m_values = np.array([np.mean(IL_data[idx][connector - 1]) for connector in connector_numbers])
        
        # Sortowanie std values zgodnie z posortowanymi connectorami
        sorted_mean_values = m_values[sorted_indices]
        
        # Rysowanie wykresu dla posortowanych danych
        ax.plot(
            range(1, len(sorted_connector_numbers) + 1),  # Indeksy na osi X
            sorted_mean_values,
            label=f'Mean (λ={wavelength} nm)',
            color=colors[idx],
            linestyle='--',
            marker='x'
        )
    
    # Ustawienie etykiet na osi X zgodnie z numerami connectorów
    ax.set_xticks(range(1, len(sorted_connector_numbers) + 1))
    ax.set_xticklabels(sorted_connector_numbers)  # Etykiety to posortowane numery connectorów
    
    # Ustawienia wykresu
    ax.set_xlabel(f'Connector Number (Ordered by Mean for λ={wavelength_to_sort} nm)')
    ax.set_ylabel('Mean of IL')
    ax.set_title('Sorted Mean of IL Values for Connectors Across Different Wavelengths')
    
    # Dodanie legendy
    ax.legend(loc='best')
    
    plt.grid(True)
    plt.tight_layout()  # Poprawienie układu wykresu
    plt.show()
    
    return fig

