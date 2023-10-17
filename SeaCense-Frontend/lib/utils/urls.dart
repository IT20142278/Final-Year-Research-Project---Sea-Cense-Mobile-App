class UrlConstants {
  // static const String environment = "development";
  static const String environment = "production";
  static const String baseUrl = "http://10.0.2.2:5000";

  static String getLiveEndpoint() {
    if (environment == "development") {
      return "$baseUrl/type-identification";
    } else {
      return "https://identifytype-w2qyntknzq-uc.a.run.app/type-identification";
    }
  }

  static String getProcessedEndpoint() {
    if (environment == "development") {
      return "$baseUrl/quality";
    } else {
      return "https://qualityprediction-w2qyntknzq-uc.a.run.app/quality";
    }
  }

  static String getPriceEndpoint() {
    if (environment == "development") {
      return "$baseUrl/price-prediction";
    } else {
      return "https://predictprice-w2qyntknzq-uc.a.run.app/price-prediction";
    }
  }

  static String getJuvenileEndpoint() {
    if (environment == "development") {
      return "$baseUrl/age-prediction";
    } else {
      return "https://predictage-w2qyntknzq-uc.a.run.app/age-prediction";
    }
  }
}
