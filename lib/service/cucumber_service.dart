import 'package:image_picker/image_picker.dart';
import 'package:sea_cense/models/base_api_response.dart';
import 'package:sea_cense/network/network.dart';

class CucumberService {
  Future<BaseAPIResponse> uploadImage(
      XFile imageFile, Function(int, int)? onSendProgress, String endpoint) async {
    BaseAPIResponse response = await Network.upload(
        filePath: imageFile.path,
        fileName: imageFile.name,
        endpoint: endpoint,
        onSendProgress: onSendProgress);
    return response;
  }
}
