class DataValidator:
    @staticmethod
    def validate_data(self, **kwargs) -> bool:
        for key, value in kwargs.items():
            if value is None:
                return False
        return True

    @staticmethod
    def get_validated_array_with_data_for_insertion(self, **kwargs) -> list:
        data_for_insertion = []
        for key, value in kwargs.items():
            data_for_insertion.append((key, value))
        return data_for_insertion