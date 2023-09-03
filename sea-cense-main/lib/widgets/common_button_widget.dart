import 'package:flutter/material.dart';

class CommonButtonWidget extends StatelessWidget {
  final String text;
  final double size;
  final Function()? onTap;
  const CommonButtonWidget({super.key, required this.text, required this.size, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: size,
        height: 60,
        alignment: Alignment.centerLeft,
        padding: const EdgeInsets.only(left: 20),
        decoration: const BoxDecoration(
          borderRadius: BorderRadius.all(Radius.circular(20)),
          image: DecorationImage(
            image: AssetImage('assets/images/common_btn_bg.png'),
            fit: BoxFit.fill,
          ),
        ),
        child: Text(
          text,
          style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w200),
        ),
      ),
    );
  }
}
