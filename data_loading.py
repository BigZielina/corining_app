
import pandas as pd
import numpy as np
import itertools as it
import xlsxwriter
import regex
import xlsxwriter.utility
class DataCore():


    """Main class for interacting with the data stored in the excel files


    ...
    Attributes
    ----
    excel_data : pd.Dataframe

    all the data from the excel file loaded into a pandas Dataframe


    """

    def __init__(self, language = 'english') -> None:
        if language not in ['english', 'polish']:
            raise Exception(f"Language {language} not supported")
        self.language = language

    def load_excel(self,path : str) -> list[str]:
        """Load the data from and excel file

        Failes if there is no excel file at the given path or
        if openpyxl is not installed

        Returns a list of string warnings if there is incorrect data in the excel file"""

        warnings = []

        excel_file = pd.ExcelFile(path)
        self.excel_data = {}
        for sheet_name in excel_file.sheet_names:

            matches = regex.findall(r"([-+]?\d*\.\d+|\d+)nm",sheet_name)

            if len(matches) <= 0:

                if sheet_name not in ["example","instruction","przykład","instrukcja"]:

                    if self.language == 'english':
                        warnings.append(f"Excel Sheet with name : {sheet_name} does not match the expected naming convention and will be ignored. Outside the 'instruction' and 'example'\
                           all excel sheets should have name in the same format as '1750nm' or '1660.1nm'.")
                    elif self.language == "polish":
                        warnings.append(f"Arkusz excela o nazwie : {sheet_name} spełnia oczekiwanej konwencji nazw . Poza arkuszami 'instrukcja' i 'przykład'\
                           nazwy wszystkich arkuszy excela powinny mieć nazwe w formacie takim jak '1750nm' or '1660.1nm'.")


                continue


            wavelength = float(matches[0])

            excel_file_df  = excel_file.parse(sheet_name,header=None)


            n_connectors = self.n_connectors(excel_file_df)
            n_fibers = self.n_fibers(excel_file_df)
            IL_data = self.all_IL_values(excel_file_df)
            expected_nan_rows_indecies = [n_connectors * i + i for i in range(n_connectors)]
            expected_nan_rows_indecies += [n_connectors * i + i + (1 if i%2==0 else -1) for i in range(n_connectors)]

            wrong_cell_values = []
            wrong_cells = []

            for row_index in expected_nan_rows_indecies:

                for column_index in range(n_fibers):

                    expected_nan_cell_value = IL_data[row_index,column_index]

                    if not np.isnan(float(expected_nan_cell_value)):

                        IL_data[row_index,column_index] = 'NaN'
                        column_letter = xlsxwriter.utility.xl_col_to_name(column_index+2)
                        wrong_cell_values.append(expected_nan_cell_value)
                        wrong_cell_values.append(f"{column_letter}{column_index+2}")

            if self.language == 'english':
                warnings.append(f"Cell value(s) {wrong_cell_values} at position(s) {wrong_cells} in sheet {sheet_name} is(are) a numeric value.\
                    The field should not contain any as it corresponds to the impossible case when a conector is matched against a connector on the same jumper.")

            elif self.language == 'polish':
                warnings.append(f"Wartości komórek(komórki) {wrong_cell_values} w pozycji(ach) {wrong_cells} w arkuszu {sheet_name} \
                    jest(są) wartościami numerycznymi. Te komórki nie powinny zawierać żadnej wartości, ponieważ odnoszą się do niemożliwej\
                    sytuacji gdzie connector jest testowany z connectorem na tym samym kablu.")


            expected_data_shape = (n_connectors*n_connectors,n_fibers)

            if IL_data.shape != expected_data_shape:

                if self.language == "english":
                    warnings.append(f"Loaded data from sheet {sheet_name} has shape {IL_data.shape} while the fiber and connector numbering suggest a shape {(n_connectors*n_connectors,n_fibers)}.")
                elif self.language == "polish":
                    warnings.append(f"Załadowane dane z arkusza {sheet_name} mają wymiary {IL_data.shape} podczas gdy numeracja connectorów i włókien sugeruje wymiary {(n_connectors*n_connectors,n_fibers)}.")

            self.excel_data[wavelength] = excel_file_df

        return warnings





    def create_excel_template(self,n_connectors,path="template.xlsx",wavelengths = [1310, 1550],n_extra_wavelengths=0,n_fibers=1):

        if n_connectors%2 != 0 :
            if self.language == "english":
                raise Exception("Connector number must be even")
            elif self.language == "polish":
                raise Exception("liczba złączek musi być parzysta")

        #add instruction and exmaple pages to the excel file
        Sheet = np.zeros((n_connectors*n_connectors+2,n_fibers+2),dtype=object)

        #setting constant cells
        if self.language == "english":
            Sheet[0,0] = "Reference Configuration"
            Sheet[1,0] = "Reference Connector"
            Sheet[0,1] = ""
            Sheet[1,1] = "DUT"
        elif self.language == "polish":
            Sheet[0,0] = "Konfiguracja Referencyjna"
            Sheet[1,0] = "Złączka referencyjna"
            Sheet[0,1] = ""
            Sheet[1,1] = "DUT"
        #setting numbering
        Sheet[1,2:n_fibers+2] = np.linspace(1,n_fibers,n_fibers)
        Sheet[2:n_connectors*n_connectors+2,1] = np.tile(np.linspace(1,n_connectors,n_connectors),n_connectors)
        Sheet[2:n_connectors*n_connectors+2,0] = np.repeat(np.linspace(1,n_connectors,n_connectors),n_connectors)


        example_sheet = Sheet.copy()
        example_sheet[2:n_connectors*n_connectors+2,2:n_fibers+2] = np.round(np.random.rand(n_connectors*n_connectors,n_fibers),3)
        excel_df = pd.DataFrame(Sheet)
        excel_df_example = pd.DataFrame(example_sheet)


        writer = pd.ExcelWriter(path,engine="xlsxwriter")

        for wavelength in wavelengths:
            excel_df.to_excel(writer,sheet_name=f"{wavelength}nm",index=False,header=False)

        for i in range(n_extra_wavelengths):
            excel_df.to_excel(writer,sheet_name=f"wavelength_{i}",index=False,header=False)

        if self.language == "english":

            excel_df_example.to_excel(writer,sheet_name=f"example",index=False,header=False)
            excel_df.to_excel(writer,sheet_name=f"instruction",index=False,header=False)

        elif self.language == "polish":
            excel_df_example.to_excel(writer,sheet_name=f"przykład",index=False,header=False)
            excel_df.to_excel(writer,sheet_name=f"instrukcja",index=False,header=False)



        workbook = writer.book
        merge_format = workbook.add_format(
            {
                "bold": 1,
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )

        nan_format = workbook.add_format({
            "bg_color" : "red"
        })
        even_format = workbook.add_format({
            "bg_color" : "gray"
        })

        for worksheet in workbook.worksheets():

            column_letter = xlsxwriter.utility.xl_col_to_name(n_fibers+1)

            worksheet.set_column("A:B",22)
            if column_letter != "C":
                if self.language == "english":
                    worksheet.merge_range(f"C1:{column_letter}1", "Fiber Number",merge_format)
                elif self.language == "polish":
                    worksheet.merge_range(f"C1:{column_letter}1", "Numer włókna",merge_format)

            else:
                if self.language == "english":
                    worksheet.write(0,2,"Fiber Number",merge_format)
                elif self.language == "polish":
                    worksheet.write(0,2,"Numer włókna",merge_format)

            for k in range(n_connectors):
                if k%2 == 0:
                    continue
                worksheet.conditional_format(n_connectors*k+2,2,n_connectors*(k+1)+1,n_fibers+1,{'type' : 'cell','criteria' : '!=','value' : '"Nan"','format' : even_format})

            for i in range(n_connectors):
                worksheet.merge_range(f"A{3+i*n_connectors}:A{2+(i+1)*n_connectors}", f"{i+1}",merge_format)


                if i%2 != 0:
                    continue
                for j in range(n_fibers):
                    worksheet.write(n_connectors*i+i+2,j+2,'NaN',nan_format)
                    worksheet.write(n_connectors*i+i+3,j+2,'NaN',nan_format)

                    worksheet.write(n_connectors*(i+1)+(i+1)+1,j+2,'NaN',nan_format)
                    worksheet.write(n_connectors*(i+1)+(i+1)+2,j+2,'NaN',nan_format)

            if worksheet.name == "instruction" or worksheet.name == "przykład":
                column_letter_start = xlsxwriter.utility.xl_col_to_name(n_fibers+5)
                column_letter_end = xlsxwriter.utility.xl_col_to_name(n_fibers+12)


                instruction_1 = f"This is a template for a test with {n_connectors//2} kablami and {n_fibers} fibers made for {wavelengths} wavelengths."+\
                "The results for each wavelength should be entered into one of the given worksheet named 'wavelength_1','wavelength_2',..."+\
                "The name of the worksheet should be changed to contain the value of the tested wavelength in the following format 'wavelength+nm'. For example '1650nm'"+\
                "The results of the test should be entered in to the cells with 0s."+\
                "Each row should contain results for all fibers from a single test.\n"

                instruction_2 = f" Column A : 'Reference connector' denotes the number of the connector currently used as a reference." +\
                "Column B : 'DUT' (Device under test) denotes the number of the connector tested against the reference connector. " +\
                f"Columns C to {column_letter} correspond to sequential fibers."

                instruction_3 = f"The red cell with 'NaN' mark impossible cases where a connector is testd against itself. No values should be entered there."+\
                f"Additionally the empty cells to the right and to the left of the table should be left empty. Entering any form of data there will result in a loading error later."

                if self.language == "polish":
                    instruction_1 = f"To jest wzór excela dla tesu z  {n_connectors//2} kablami i {n_fibers} włókanmi dla światło o długosciach {wavelengths}."+\
                    "Wyniki dla każdej długści fali powinny być wstawiane  danego arkusza o nazwie 'wavelength_1','wavelength_2',..."+\
                    "Nazwa arkusza powinna zawierać długość fali w takim formacie 'wavelength+nm'. Na przykład '1650nm'"+\
                    "Wyniki każdego testu powinny być wstawiane do komórek z zerami."+\
                    "Każdy rząd powinnien zawierać wyniki dla wszystkich włókien z danego testu.\n"

                    instruction_2 = f" Kolumna A : 'Złączka referencyjna' oznacza numer złączki używanej jako odniesienia dla innych" +\
                    "Kolumna B : 'DUT' (Device under test) oznacza numer złączki obecnie testowanej" +\
                    f"Kolumny C do {column_letter} odpowiadają numerom kolejnych włókien."

                    instruction_3 = f"Czerwone komórki z wartościami 'NaN' zaznaczają niemożliwe przypadki gdzie złączka jest testowana ze złączką na tym samym kablu. Powinny one pozostać niezmienione."+\
                    f"Dodatkowo puste pola dookoła miejsca na dane powinny pozostać puste. Wprowadzanie tam danych może spowodować błędy."

                merge_format.set_text_wrap()
                worksheet.merge_range(f"{column_letter_start}{5}:{column_letter_end}{12}", instruction_1,merge_format)
                worksheet.merge_range(f"{column_letter_start}{14}:{column_letter_end}{19}", instruction_2,merge_format)
                worksheet.merge_range(f"{column_letter_start}{21}:{column_letter_end}{25}", instruction_3,merge_format)




        writer.close()


    def all_cells(self,data : pd.DataFrame) -> np.ndarray:
        """Returns values all cells read from a given DataFrame

        Since data from the excel file is stored in a dictionary this should
        be called on output of DataCore.IL_wavelength"""
        return data.to_numpy()


    def all_IL_values(self,data : pd.DataFrame) -> np.ndarray:
        """"Returns values of all cells where we expect IL values"""
        n_fibers = self.n_fibers(data)
        n_connectors = self.n_connectors(data)
        return self.all_cells(data)[2:n_connectors*n_connectors+2,2:n_fibers+2]

    def n_connectors(self,data : pd.DataFrame) -> int:
        """Returns the expected number of connectors present in the data"""
        #the second dimensions corresponds to the number of connectors
        # the first one cannot be used since it contains multiple entries
        # for different wavelengths
        connector_number_index = 2
        previous_cell = self.all_cells(data)[connector_number_index,1]
        while True:

            if self.all_cells(data)[connector_number_index+1,1] != previous_cell + 1:
                break
            previous_cell = self.all_cells(data)[connector_number_index + 1,1]
            connector_number_index += 1

        return int(self.all_cells(data)[connector_number_index,1])

    def n_fibers(self,data : pd.DataFrame) -> int:
        """Returns the expected number of fiber present in the data"""
        #the second dimensions corresponds to the number of connectors
        # the first one cannot be used since it contains multiple entries
        # for different wavelengths
                # for different wavelengths
        fiber_number_index = 2
        previous_cell = self.all_cells(data)[1,fiber_number_index]
        for fiber_number_index in range(2,self.all_cells(data).shape[1]-1):
            if self.all_cells(data)[1,fiber_number_index+1] != previous_cell + 1:
                break
            previous_cell = self.all_cells(data)[1,fiber_number_index + 1]
            fiber_number_index += 1

        return int(previous_cell)

    def n_jumpers(self,data : pd.DataFrame) -> int:
        """Returns the expected number of jumpers present in the data"""

        # there are two connectors per jumper
        return self.n_connectors(data)//2


    def wavelengths(self) -> list[float]:
        """Returns a list of wavelenghs found in the data"""

        return list(self.excel_data.keys())


    def IL_reference_connetor(self, data : pd.DataFrame, index : int) -> np.ndarray:
        """Returns values of IL for a given reference connector"""

        return self.all_IL_values(data)[:,index*self.n_connectors():(index+1)*self.n_connectors()]


    def IL_reference_jumper(self, data : pd.DataFrame, index : int) -> np.ndarray:
        """Returns values of IL for a given reference jumper"""
        return self.all_IL_values(data)[:,2*index*self.n_connectors():2*(index+1)*self.n_connectors()]



    def IL_wavelength(self,wavelength : float) -> pd.DataFrame:
        """Returns values of IL for a given wavelength"""


        return self.excel_data[wavelength]


    def filter_n_jumper(self,IL_data : np.ndarray,n_connectors : int, jumper_indecies : list[int]) -> np.ndarray:
        """Filters the data to only include the data for a given jumper indecies

        Works only on data of shape m^2 x n. In practive this means using data from
        DataCore.IL_wavelength
        """


        connector_indecies = []
        for jumper_index in jumper_indecies:
            connector_indecies.append(2*jumper_index)
            connector_indecies.append(2*jumper_index+1)

        excel_indecies = []
        for index_reference in connector_indecies:
            for index_dut in connector_indecies:
                excel_indecies.append(index_reference*n_connectors+index_dut)

        return IL_data[excel_indecies,:]


    def jumper_combinations(self, IL_data : pd.DataFrame, n_choices : int) -> list[np.ndarray]:

        if n_choices > self.n_jumpers(IL_data):
            print(f"Cannot chose {n_choices} from {self.n_jumpers(IL_data)}.")

        all_combinations_tupples = list(it.combinations(range(0,self.n_jumpers(IL_data)),n_choices))
        if len(all_combinations_tupples) > 1e5:
            print("WARN! Number of combinations is larger than 10 000! This will have a huge impact on performance")

        IL_data_combinations = []


        IL_data_numpy = self.all_IL_values(IL_data)
        for combination in all_combinations_tupples:

            data = self.filter_n_jumper(IL_data_numpy,self.n_connectors(IL_data),list(combination))

            IL_data_combinations.append(data)

        return IL_data_combinations

    def jumper_combinations_all_wavelengths(self, n_choices : int) -> dict[float, list[np.ndarray]]:


        wavelength_IL_combinations = {}
        for wavelength in self.wavelengths():

            IL_data_wavelength = self.IL_wavelength(wavelength)
            IL_data_combinations = self.jumper_combinations(IL_data_wavelength,n_choices)
            wavelength_IL_combinations[wavelength] = IL_data_combinations


        return wavelength_IL_combinations

    def map_dict(self, func, d : dict) -> dict:
        """Map a function over a dictionary

        For a dictionary {a : x, b : y, ...} and a function f
        this function returns {a : f(x), b : f(y), ...}"""
        return dict(map(lambda x : (x,func(d[x])),d))


    def IL_wavelengths(self) -> dict[float,np.ndarray]:
        return {wavelength : self.all_IL_values(self.IL_wavelength(wavelength)) for wavelength in self.wavelengths()}


    def IL_reference_connectors(self) ->  np.ndarray:
        """Create a list containg all IL values for each of the reference connectors

        Returns an array of shape n_connectors x n_wavelengths*n_connectors x n_fibers """
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = np.split(self.all_IL_values(data),self.n_connectors(data),axis=0)
            connector_data.append(np.array(connector))

        return np.hstack(connector_data)

    def split_array(self,arr, k):
        split_arrays = []
        for i in range(k):
            split_arrays.append(arr[i::k])
        return split_arrays

    def IL_dut_connectors(self) -> np.ndarray:
        """Create a list containg all IL values for each of the DUT connectors

        Returns an array of shape n_connectors x n_wavelengths*n_connectors x n_fibers.
        Should fullfill the same job as DataCore.IL_reference_connectors"""
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = self.split_array(self.all_IL_values(data),self.n_connectors(data))
            connector_data.append(np.array(connector))

        return np.hstack(connector_data)

    def IL_jumpers_wavelengths(self) -> np.ndarray:
        """Create a list containg all IL values for each of the jumpers for each wavelength

        Returns an dict with arrays of shape 2*n_connectors x n_fibers."""
        wave_IL = self.excel_data
        jumper_data = {}
        for wave in wave_IL:
            data = wave_IL[wave]
            jumper = np.split(self.all_IL_values(data),self.n_jumpers(data))
            jumper_data[wave] = jumper

        return jumper_data

    def IL_fibers(self) -> np.ndarray:
        """Create a list containg all IL values for each of fiber

        Returns an array of shape n_fibers x n_wavelengths*n_connectors*n_connectors x 1.
        Could be flattened but it does not impact aggregate functions like mean, std, etc."""
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = np.split(self.all_IL_values(data),self.n_fibers(data),axis=1)
            connector_data.append(np.array(connector))

        return np.hstack(connector_data)


    def filter_nan(self,A : np.ndarray) -> np.ndarray:
        """Remove NaN values from ndarray or list of ndarray"""
        if type(A) == type([]):
            float_cast = [ a.astype(float) for a in A]
            return np.array([ a[~np.isnan(a)] for a in float_cast])

        float_cast = A.astype(float)
        return float_cast[~np.isnan(float_cast)]

def generate_df(file_path):

    #example
    DC = DataCore()
    DC.load_excel(file_path)


    wavelength_ex = DC.wavelengths()[0]

    test_sheet = DC.IL_wavelength(wavelength_ex)

    #-----------------------------------------------------------------

    num_connectors = DC.n_connectors(test_sheet)
    num_jumpers = DC.n_jumpers(test_sheet)
    num_fibers = DC.n_fibers(test_sheet)

    print(f"Number of connectors {DC.n_connectors(test_sheet)}")
    print(f"Number of jumper {DC.n_jumpers(test_sheet)}")
    print(f"Number of fiber {DC.n_fibers(test_sheet)}")


    data1 = {
        "Number of Connectors" : [num_connectors],
        "Number of Jumpers" : [num_jumpers],
        "Number of Fibers" : [num_fibers]
    }

    df1 = pd.DataFrame(data1)

    return (df1,) , num_jumpers
