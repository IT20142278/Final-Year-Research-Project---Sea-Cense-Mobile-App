class BaseAPIResponse {
  BaseAPIResponse({this.status, this.data, required this.error});

  bool error;
  int? status;
  dynamic data;
}
