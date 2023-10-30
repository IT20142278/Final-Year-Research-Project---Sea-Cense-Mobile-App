import 'package:dio/dio.dart';
import 'package:sea_cense/models/base_api_response.dart';

class Network {
  static Future<BaseAPIResponse> upload(
      {required String filePath,
      required String fileName,
      required String endpoint,
      required Function(int, int)? onSendProgress}) async {
    FormData data = FormData.fromMap({
      "image_path": await MultipartFile.fromFile(
        filePath,
        filename: fileName,
      ),
    });

    Dio dio = Dio();
    dio.options.connectTimeout = const Duration(seconds: 20);

    try {
      var response = await dio.post(
        endpoint,
        data: data,
        onSendProgress: onSendProgress,
      );

      return BaseAPIResponse(data: response.data, error: false, status: response.statusCode);
    } on DioException catch (e) {
      return BaseAPIResponse(data: null, error: true, status: e.response?.statusCode);
    }
  }
}
