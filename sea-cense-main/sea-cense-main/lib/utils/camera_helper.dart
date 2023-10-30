import 'package:image_picker/image_picker.dart';

class CameraHelper {
  static Future<XFile> selectImages() async {
    final XFile? imageFile;
    final ImagePicker picker = ImagePicker();
    imageFile = await picker.pickImage(imageQuality: 85, source: ImageSource.gallery);

    return imageFile!;
  }

  static Future<XFile> takeImages() async {
    final XFile? imageFile;
    final ImagePicker picker = ImagePicker();
    imageFile = await picker.pickImage(imageQuality: 85, source: ImageSource.camera);

    return imageFile!;
  }
}
